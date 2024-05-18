from transformers import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
    DisjunctiveConstraint,
    BeamSearchScorer,
    PhrasalConstraint,
    ConstrainedBeamSearchScorer,
    PreTrainedModel,
)
import numpy as np
import random
import warnings
import inspect

from transformers.generation import validate_stopping_criteria, EosTokenCriteria
from transformers.generation.utils import GenerateOutput, SampleOutput, logger
import torch
from typing import Callable, List, Optional, Union
from torch import nn
import torch.distributed as dist
import copy


class NewGenerationMixin(GenerationMixin):
    @torch.no_grad()
    def generate_(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        # self._validate_model_kwargs(model_kwargs.copy())

        if kwargs.get("do_stream", False):
            # 2. Set generation parameters if not already defined
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

            if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
                if model_kwargs.get("attention_mask", None) is None:
                    logger.warning(
                        "The attention mask and the pad token id were not set. As a consequence, you may observe "
                        "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                    )
                eos_token_id = generation_config.eos_token_id
                if isinstance(eos_token_id, list):
                    eos_token_id = eos_token_id[0]
                logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
                generation_config.pad_token_id = eos_token_id

            # 3. Define model inputs
            # inputs_tensor has to be defined
            # model_input_name is defined if model-specific keyword input is passed
            # otherwise model_input_name is None
            # all model-specific keyword inputs are removed from `model_kwargs`
            inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
                inputs, generation_config.bos_token_id, model_kwargs
            )
            batch_size = inputs_tensor.shape[0]

            # 4. Define other model kwargs
            model_kwargs["output_attentions"] = generation_config.output_attentions
            model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
            if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
                model_kwargs["use_cache"] = True
            else:
                model_kwargs["use_cache"] = generation_config.use_cache

            accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
            requires_attention_mask = "encoder_outputs" not in model_kwargs

            if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
                model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                    inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
                )

            # decoder-only models should use left-padding for generation
            if not self.config.is_encoder_decoder:
                if (
                    generation_config.pad_token_id is not None
                    and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id)
                    > 0
                ):
                    logger.warning(
                        "A decoder-only architecture is being used, but right-padding was detected! For correct "
                        "generation results, please set `padding_side='left'` when initializing the tokenizer."
                    )

            if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
                # if model is encoder decoder encoder_outputs are created
                # and added to `model_kwargs`
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, model_kwargs, model_input_name
                )

            # 5. Prepare `input_ids` which will be used for auto-regressive generation
            if self.config.is_encoder_decoder:
                input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                    batch_size=batch_size,
                    model_input_name=model_input_name,
                    model_kwargs=model_kwargs,
                    decoder_start_token_id=generation_config.decoder_start_token_id,
                    bos_token_id=generation_config.bos_token_id,
                    device=inputs_tensor.device,
                )
            else:
                input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

            # 6. Prepare `max_length` depending on other stopping criteria.
            input_ids_length = input_ids.shape[-1]
            has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
            has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
            generation_config = self._prepare_generated_length(
                generation_config=generation_config,
                has_default_max_length=has_default_max_length,
                has_default_min_length=has_default_min_length,
                model_input_name=model_input_name,
                inputs_tensor=inputs_tensor,
                input_ids_length=input_ids_length,
            )

            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            prepared_logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
            )
            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 9. prepare stopping criteria
            stopping_criteria = self._get_stopping_criteria(
                generation_config=generation_config, stopping_criteria=stopping_criteria
            )

            # 13. run sample
            return self.sample_stream(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )
        else:
            if model_kwargs.get('do_stream'):
                model_kwargs.pop('do_stream')
            return self.generate(
                inputs=inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                assistant_model=assistant_model,
                streamer=streamer,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                **model_kwargs
            )

    @torch.no_grad()
    def sample_stream(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""

        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            # logger.warning_once(
            #     "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
            #     " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
            #     " Otherwise make sure to set `model.generation_config.eos_token_id`",
            #     FutureWarning,
            # )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        # auto-regressive generation
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            yield next_tokens
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            # if eos_token_id is not None:
            #     unfinished_sequences = unfinished_sequences.mul(
            #         (sum(next_tokens != i for i in eos_token_id)).long()
            #     )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0:
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True


def init_stream_support():
    PreTrainedModel.generate_ = NewGenerationMixin.generate_
    PreTrainedModel.sample_stream = NewGenerationMixin.sample_stream