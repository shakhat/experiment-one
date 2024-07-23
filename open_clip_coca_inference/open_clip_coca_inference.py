import argparse
import time
from typing import Optional
import numpy as np
import open_clip
import torch
from PIL import Image
import onnxruntime as ort


# copied from coca_model.py
from torch.nn import functional as F
from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    TopPLogitsWarper,
    TopKLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    MinLengthLogitsProcessor,
    MaxLengthCriteria,
    StoppingCriteriaList
)
from open_clip.transformer import _expand_token


def coca_forward(  # from CoCa model forward
        runtime,
        model,
        onnx_text_generator,
        text: Optional[torch.Tensor],
        image_embs: Optional[torch.Tensor],
):
    text_transformer = model.text
    
    # copied from TextTransformer.forward()
    cast_dtype = text_transformer.transformer.get_cast_dtype()
    seq_len = text.shape[1]

    x = text_transformer.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
    attn_mask = text_transformer.attn_mask
    if text_transformer.cls_emb is not None:
        seq_len += 1
        x = torch.cat([x, _expand_token(text_transformer.cls_emb, x.shape[0])], dim=1)
        # in the open_clip implementation the attention mask is changed to 3d (per 1 layer) 
        # with added mask that hides padding tokens; but this is not needed since the text 
        # can not have any padding tokens because that would mean we ran beyond the end 
        # cls_mask = text_transformer.build_cls_mask(text, cast_dtype)
        # if attn_mask is not None:
        #     attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

    x = x + text_transformer.positional_embedding[:seq_len].to(cast_dtype)
    
    # print(f"x.shape = {x.shape}")
    # print(f"attention_mask.shape = {attn_mask.shape}")

    attn_mask = attn_mask[:seq_len, :seq_len]
    if runtime == "onnx":
        model_inputs = onnx_text_generator.get_inputs()
        outputs = onnx_text_generator.run(None, {
            model_inputs[0].name: x.numpy(), 
            model_inputs[1].name: attn_mask.numpy(), 
            model_inputs[2].name: image_embs.numpy(), 
            })
        x = torch.tensor(outputs[0])
        return x
    else:
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = text_transformer.transformer(x, attn_mask=attn_mask)
        
        # print(f"x.shape = {x.shape}")
        x = x.permute(1, 0, 2)  # LND -> NLD (cause MultimodalTransformer expects NLD)

        # presence of appended cls embed (CoCa) overrides pool_type, always take last token
        token_embs = x[:, :-1]  # _, tokens = text_global_pool(x, pool_type='last')
        
        # MultimodalTransformer cuts attention mask to match the sequence length, so we don't need to worry about this
        logits = model.text_decoder.forward(image_embs, token_embs)  # MultimodalTransformer
        
        # print(f"text.shape = {text.shape}, token_embs.shape = {token_embs.shape}, logits.shape = {logits.shape}")
        return logits


def _generate_beamsearch(
        runtime,
        onnx_text_generator,
        image_inputs,
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        stopping_criteria=None,
        logit_processor=None,
):
    device = image_inputs.device
    batch_size = image_inputs.shape[0]
    image_inputs = torch.repeat_interleave(image_inputs, num_beams, dim=0)
    
    start = time.perf_counter_ns()
    _, image_embs = model._encode_image(image_inputs)
    print(f"Image encoding time: {(time.perf_counter_ns() - start) / 10**6} ms")
    print(f"image_embs.shape = {image_embs.shape}")
    
    input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
    input_ids = input_ids * sot_token_id
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=device,
        num_beam_groups=num_beam_groups,
    )

    num_beams = beam_scorer.num_beams
    num_beam_groups = beam_scorer.num_beam_groups
    num_sub_beams = num_beams // num_beam_groups
    batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
    batch_beam_size, cur_len = input_ids.shape
    beam_indices = None

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
    # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
    # the same group don't produce same tokens everytime.
    beam_scores[:, ::num_sub_beams] = 0
    beam_scores = beam_scores.view((batch_size * num_beams,))

    while True:

        # predicted tokens in cur_len step
        current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

        # indices which will form the beams in the next time step
        reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

        # do one decoder step on all beams of all sentences in batch
        logits = coca_forward(runtime, model, onnx_text_generator, input_ids, image_embs=image_embs)

        for beam_group_idx in range(num_beam_groups):
            group_start_idx = beam_group_idx * num_sub_beams
            group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
            group_size = group_end_idx - group_start_idx

            # indices of beams of current group among all sentences in batch
            batch_group_indices = []

            for batch_idx in range(batch_size):
                batch_group_indices.extend(
                    [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                )
            group_input_ids = input_ids[batch_group_indices]

            # select outputs of beams of currentg group only
            next_token_logits = logits[batch_group_indices, -1, :]
            vocab_size = next_token_logits.shape[-1]

            next_token_scores_processed = logit_processor(
                group_input_ids, next_token_logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx
            )
            next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
            next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

            # reshape for beam search
            next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
            beam_outputs = beam_scorer.process(
                group_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=process_beam_indices,
                group_index=beam_group_idx,
            )
            beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids[batch_group_indices] = group_input_ids[beam_idx]
            group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            current_tokens[batch_group_indices] = group_input_ids[:, -1]

            # (beam_idx // group_size) -> batch_idx
            # (beam_idx % group_size) -> offset of idx inside the group
            reordering_indices[batch_group_indices] = (
                num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
            )

        input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

        # increase cur_len
        cur_len = cur_len + 1
        if beam_scorer.is_done or stopping_criteria(input_ids, None):
            break

    final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=final_beam_indices,
    )
    return sequence_outputs['sequences']


def generate(
    runtime, 
    model,
    onnx_image_transformer,
    onnx_text_generator,
    image,
    text=None,
    seq_len=30,
    max_seq_len=77,
    temperature=1.,
    generation_type="beam_search",
    top_p=0.1,  # keep tokens in the 1 - top_p quantile
    top_k=1,  # keeps the top_k most probable tokens
    pad_token_id=None,
    eos_token_id=None,
    sot_token_id=None,
    num_beams=6,
    num_beam_groups=3,
    min_seq_len=5,
    stopping_criteria=None,
    repetition_penalty=1.0,
    fixed_output_length=False # if True output.shape == (batch_size, seq_len)
):
    with torch.no_grad():
        sot_token_id = 49406 if sot_token_id is None else sot_token_id
        eos_token_id = 49407 if eos_token_id is None else eos_token_id
        pad_token_id = model.pad_id if pad_token_id is None else pad_token_id
        logit_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                RepetitionPenaltyLogitsProcessor(repetition_penalty),
            ]
        )

        if stopping_criteria is None:
            stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

        stopping_criteria = StoppingCriteriaList(
            stopping_criteria
        )

        device = image.device

        if generation_type in "beam_search":
            output = _generate_beamsearch(
                runtime, 
                onnx_text_generator,
                image_inputs=image,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                sot_token_id=sot_token_id,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                stopping_criteria=stopping_criteria,
                logit_processor=logit_processor,
            )
            if fixed_output_length and output.shape[1] < seq_len:
                return torch.cat(
                    (output, torch.ones(output.shape[0], seq_len-output.shape[1], device=device, dtype=output.dtype) * pad_token_id),
                    dim=1
                )
            return output

        elif generation_type == "top_p":
            logit_warper = TopPLogitsWarper(top_p)
        elif generation_type == "top_k":
            logit_warper = TopKLogitsWarper(top_k)
        else:
            raise ValueError(f"unknown generation_type {generation_type}")

        start = time.perf_counter_ns()
        if runtime == "onnx":
            img_data = image.numpy().astype(np.float32)
            model_inputs = onnx_image_transformer.get_inputs()
            outputs = onnx_image_transformer.run(None, {model_inputs[0].name: img_data})
            image_embs = torch.tensor(outputs[1])
        else:
            _, image_embs = model._encode_image(image)
            
        print(f"Image encoding time: {(time.perf_counter_ns() - start) / 10**6} ms")

        if text is None:
            text = torch.ones((image.shape[0], 1), device=device, dtype=torch.long) * sot_token_id

        num_dims = len(text.shape)

        if num_dims == 1:
            text = text[None, :]

        cur_len = text.shape[1]
        model.eval()
        out = text

        while True:
            x = out[:, -max_seq_len:]
            cur_len = x.shape[1]
            mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
            sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

            if mask.all():
                if not fixed_output_length:
                    break
            else:
                logits = coca_forward(runtime, model, onnx_text_generator, x, image_embs=image_embs)[:, -1]
                logits = logits[~mask, :]
                filtered_logits = logit_processor(x[~mask, :], logits)
                filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

                if (cur_len + 1 == seq_len):
                    sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                else:
                    sample[~mask, :] = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            
            # print(f"cur_len = {cur_len}, sample = {sample}")

            cur_len += 1

            if stopping_criteria(out, None):
                break

        if num_dims == 1:
            out = out.squeeze(0)

        return out

# end of copy

def run(runtime, device, img_name, seq_len, generation_type, verbose):
    if device.startswith("npu"):
        try:
            import torch_npu  # must go after cv2 otherwise "ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block"
        except ImportError:
            raise RuntimeError("'torch-npu' library is required for Ascend NPU devices")
    print(f"Runtime: {runtime}, device: {device}, generation type: {generation_type}, max result length: {seq_len}")
    
    pt_device = torch.device(device)
    device_index = pt_device.index or 0
    if device.startswith("npu"):
        providers = [
            ("CANNExecutionProvider", {"device_id": device_index,},),
        ]
    elif device.startswith("cuda"):
        providers = [
            ("CUDAExecutionProvider", {"device_id": device_index,},),
        ]
    else:
        providers = ["CPUExecutionProvider"]
        
    options = ort.SessionOptions()
    if verbose:
        options.log_severity_level = 0
        options.log_verbosity_level = 0

    if runtime == "onnx":
        pass
    else:
        model.to(pt_device)

    # Encode image
    onnx_image_transformer = None
    if runtime == "onnx":    
        onnx_image_transformer = ort.InferenceSession(image_transformer_onnx_file, sess_options=options, providers=providers)
    else:
        model.to(pt_device)

    # Preprocess the image data
    image = preprocess(Image.open(img_name)).unsqueeze(0)

    onnx_text_generator = None
    if runtime == "onnx":
        onnx_text_generator = ort.InferenceSession(text_generator_onnx_file, sess_options=options, providers=providers)
    else:
        image = image.to(pt_device)    

    start = time.perf_counter_ns()
    with torch.no_grad():
        generated = generate(runtime, model, onnx_image_transformer, onnx_text_generator, image, generation_type=generation_type, seq_len=seq_len)

    output = generated[0]  
    print(open_clip.decode(output).split("<end_of_text>")[0].replace("<start_of_text>", ""))
    print(f"Text generation time: {(time.perf_counter_ns() - start) / 10**6} ms")


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for tensor operations.")
    parser.add_argument("--runtime", type=str, default="pytorch", choices=["onnx", "pytorch"], help="A runtime to use.")
    parser.add_argument("--len", type=int, default=30, help="A max length of a generated text.")
    parser.add_argument("--generation_type", type=str, default="top_k", choices=["beam_search", "top_k", "top_p"], help="Algorithm to select the best generated output.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging.")
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms(
      model_name="coca_ViT-L-14",
      pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    model.eval()
    
    # ONNX model files
    image_transformer_onnx_file = "open_clip_coca_image_transformer.onnx"
    text_generator_onnx_file = "open_clip_coca_text_generator.onnx"
    
    run(args.runtime, args.device, args.img, args.len, args.generation_type, args.verbose)
