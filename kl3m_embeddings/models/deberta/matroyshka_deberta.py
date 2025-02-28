"""
Wrapped DebertaV2 model that overrides the .forward() method with an additional
`reduced_dim` parameter to support training "frontloaded" or hierarchical models.

This is vendored from transformers/models/deberta_v2/modeling_deberta_v2.py.
"""
# leave huggingface code alone for comparison
# type: ignore
# pylint: disable=no-else-raise,too-many-ancestors,too-many-arguments,too-many-positional-arguments

# imports
from typing import Optional, Tuple, Union

# packages
import torch
from torch.nn import CrossEntropyLoss

# project
from transformers import DebertaV2ForMaskedLM, DebertaV2Model
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2OnlyMLMHead


class MatroyshkaDebertaV2Model(DebertaV2Model):
    """
    DebertaV2 model with an additional `reduced_dim` parameter to support training
    of a Matroyshka-style model.
    """

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        reduced_dim: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # check that reduced_dim is >0 and <hidden_size if not none
        if reduced_dim is not None:
            if reduced_dim <= 0 or reduced_dim >= self.config.hidden_size:
                raise ValueError("reduced_dim must be >0 and <hidden_size")

        device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # if reduced dim is provided, then zero the elements after reduced_dim in the embedding_output
        if reduced_dim is not None:
            embedding_output[:, :, reduced_dim:] = 0

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[
                (1 if output_hidden_states else 2) :
            ]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states
            if output_hidden_states
            else None,
            attentions=encoder_outputs.attentions,
        )


class MatroyshkaDebertaV2ForMaskedLM(DebertaV2ForMaskedLM):
    """
    DebertaV2 model with an additional `reduced_dim` parameter to support training
    of a Matroyshka-style model.
    """

    def __init__(self, config):
        """
        Initialize the model by calling the parent class and initializing the
        model with our base DebertaV2 model and then adding the standard MLM head.

        Args:
            config:
        """
        super().__init__(config)

        self.deberta = MatroyshkaDebertaV2Model(config)
        self.cls = DebertaV2OnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        reduced_dim: Optional[int] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        reduced_dim (`int`, *optional*): The reduced dimensionality of the input embeddings to use
          for training a "Matroyshka" model.
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            reduced_dim=reduced_dim,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
