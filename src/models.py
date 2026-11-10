import open_clip
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPConfig,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
)
from sentence_transformers import SentenceTransformer
import importlib


def auto_determine_dtype():
    """automatic dtype setting. override this if you want to force a specific dtype"""
    compute_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    torch_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    print(f"compute_dtype:\t{compute_dtype}")
    print(f"torch_dtype:\t{torch_dtype}")
    return compute_dtype, torch_dtype


def check_bfloat16_support():
    """checks if cuda driver/device supports bfloat16 computation"""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(current_device)
        if compute_capability[0] >= 7:  # Check if device supports bfloat16
            return True
        else:
            return False
    else:
        return None


def load_llm(llm_model_path, qlora=False, force_download=False, from_init=False):
    """load huggingface language model"""
    compute_dtype, torch_dtype = auto_determine_dtype()

    quantization_config = None
    """
    if qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    """
    if from_init:
        config = AutoConfig.from_pretrained(
            llm_model_path,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            force_download=force_download,
            output_hidden_states=True,
        )
        language_model = AutoModelForCausalLM.from_config(config)
        language_model = language_model.to(torch_dtype)
        language_model = language_model.to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        language_model = language_model.eval()

    else:
        language_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            force_download=force_download,
            output_hidden_states=True,
            trust_remote_code=True,
        ).eval()

    return language_model


def load_tokenizer(llm_model_path):
    """setting up tokenizer. if your tokenizer needs special settings edit here."""
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

    if "huggyllama" in llm_model_path:
        tokenizer.pad_token = "[PAD]"
    else:
        # pass
        # tokenizer.add_special_tokens({"pad_token":"<pad>"})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14",
        device="cuda",
        random_init=False,
    ):
        super().__init__()

        self.device = device

        # Load model
        if random_init:
            config = CLIPConfig.from_pretrained(model_name)
            self.model = (CLIPModel(config).to(device))
        else:
            self.model = (CLIPModel.from_pretrained(model_name).to(device))

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        self.text_transformer = self.model.text_model
        self.text_projection = self.model.text_projection

        # Get context length and determine chunk size
        self.max_length = (self.model.config.text_config.max_position_embeddings)
        self.num_special_tokens = (self.tokenizer.num_special_tokens_to_add(pair=False))
        self.chunk_size = (self.max_length - self.num_special_tokens)

        self.model.eval()

    def _chunk_text(self, text):
        """
        Tokenize text without special tokens and split it into
        chunks that leave room for BOS/EOS.
        """
        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            max_length=None,
        )["input_ids"]

        chunks = [
            token_ids[i:i + self.chunk_size]
            for i in range(
                0,
                len(token_ids),
                self.chunk_size,
            )
        ]

        return chunks

    def forward(self, texts):

        # Create chunks
        all_chunks = []
        chunk_to_text = []

        for text_idx, text in enumerate(texts):

            chunks = self._chunk_text(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_to_text.append(text_idx)

        # Add BOS/EOS and pad
        input_ids = []

        for chunk in all_chunks:

            chunk = (
                [self.tokenizer.bos_token_id]
                + chunk
                + [self.tokenizer.eos_token_id]
            )

            padding_length = (
                self.max_length - len(chunk)
            )

            chunk = (
                chunk
                + [self.tokenizer.pad_token_id]
                * padding_length
            )

            input_ids.append(chunk)

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        )

        # Attention mask
        attention_mask = (
            input_ids != self.tokenizer.pad_token_id
        ).long()

        # Content mask (for determining chunk lengths/weights)
        content_mask = (
            (input_ids != self.tokenizer.pad_token_id)
            & (input_ids != self.tokenizer.bos_token_id)
            & (input_ids != self.tokenizer.eos_token_id)
        )

        # Model forward
        outputs = self.text_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Hidden states
        hidden_states = torch.stack(outputs["hidden_states"][1:]).permute(1, 0, 2, 3)

        # Mean-pool hidden states within each chunk
        mask = (attention_mask.unsqueeze(1).unsqueeze(-1))
        hidden_states = ((hidden_states * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1))

        # Projected representation
        pooler_output = outputs.pooler_output
        projected = self.text_projection(pooler_output)

        # Combine chunks belonging to each caption
        batch_size = len(texts)

        final_hidden_states = []
        final_projected = []

        for text_idx in range(batch_size):

            # Find chunks belonging to this caption
            chunk_indices = [
                i
                for i, idx in enumerate(chunk_to_text)
                if idx == text_idx
            ]

            chunk_indices = torch.tensor(
                chunk_indices,
                device=self.device,
                dtype=torch.long,
            )

            # Weight chunks by number of content tokens
            chunk_lengths = (content_mask[chunk_indices].sum(dim=1).to(hidden_states.dtype))
            weights = (chunk_lengths / chunk_lengths.sum().clamp(min=1))

            # Combine hidden states
            caption_hidden_states = (hidden_states[chunk_indices] * weights[:, None, None]).sum(dim=0)

            # Combine projected representations
            caption_projected = (projected[chunk_indices] * weights[:, None]).sum(dim=0)

            final_hidden_states.append(caption_hidden_states)
            final_projected.append(caption_projected)

        # Stack back into batch
        hidden_states = torch.stack(final_hidden_states, dim=0)
        projected = torch.stack(final_projected, dim=0)

        return hidden_states, projected


class CLIPVisionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda", random_init=False):
        super().__init__()
        self.device = device
        if random_init:
            config = CLIPConfig.from_pretrained(model_name)
            self.model = CLIPModel(config).to(device)
        else:
            self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.vision_transformer = self.model.vision_model
        self.vision_projection = self.model.visual_projection

    def forward(
        self,
        images,
    ):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"]

        outputs = self.vision_transformer(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        hidden_states = torch.stack(outputs["hidden_states"][1:]).permute(1, 0, 2, 3)
        hidden_states = hidden_states[:, :, 0, :]

        pooler_output = outputs.pooler_output
        projected = self.vision_projection(pooler_output)

        return hidden_states, projected


class SiglipVisionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda", random_init=False):
        super().__init__()
        self.device = device
        if random_init:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config).vision_model.to(device)
        else:
            self.model = AutoModel.from_pretrained(model_name).vision_model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def forward(
        self,
        images,
    ):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"]

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = torch.stack(outputs["hidden_states"][1:]).permute(1, 0, 2, 3)
        hidden_states = hidden_states[:, :, :].mean(dim=2)

        projected = outputs["pooler_output"]

        return hidden_states, projected


class SiglipTextEncoder(nn.Module):
    def __init__(
        self,
        model_name="google/siglip2-so400m-patch14-224",
        device="cuda",
        random_init=False,
        max_length=64,
    ):
        super().__init__()

        self.device = device
        self.max_length = max_length

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

        # Load model
        if random_init:
            config = AutoConfig.from_pretrained(model_name)
            self.model = (AutoModel.from_config(config).text_model.to(device))
        else:
            self.model = (AutoModel.from_pretrained(model_name).text_model.to(device))

        self.model.eval()

        # Determine chunk size
        self.num_special_tokens = (self.tokenizer.num_special_tokens_to_add(pair=False))
        self.chunk_size = (self.max_length - self.num_special_tokens)

    def _chunk_text(self, text):
        """
        Tokenize text without special tokens and split into
        chunks that leave room for special tokens.
        """

        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        chunks = [
            token_ids[i:i + self.chunk_size]
            for i in range(
                0,
                len(token_ids),
                self.chunk_size,
            )
        ]

        return chunks

    def forward(self, texts):

        # Create all chunks
        all_chunks = []
        chunk_to_text = []

        for text_idx, text in enumerate(texts):

            chunks = self._chunk_text(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_to_text.append(text_idx)

        # Create padded batch of token IDs
        input_ids = []

        for chunk in all_chunks:

            # Add EOS
            chunk = chunk + [
                self.tokenizer.eos_token_id
            ]

            # Pad to max_length
            padding_length = (
                self.max_length - len(chunk)
            )

            chunk = (
                chunk
                + [self.tokenizer.pad_token_id]
                * padding_length
            )

            input_ids.append(chunk)

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        )

        # Model forward
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        # Hidden states
        hidden_states = torch.stack(outputs["hidden_states"][1:]).permute(1, 0, 2, 3)

        mask = ((input_ids != self.tokenizer.pad_token_id) & (input_ids != self.tokenizer.eos_token_id))
        mask = mask.unsqueeze(1).unsqueeze(-1)

        # Mean-pool tokens within each chunk
        hidden_states = ((hidden_states * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1))

        # Projected representation
        projected = outputs["pooler_output"]

        # Combine chunks belonging to the same caption
        batch_size = len(texts)

        final_hidden_states = []
        final_projected = []

        for text_idx in range(batch_size):

            chunk_indices = [
                i
                for i, idx in enumerate(chunk_to_text)
                if idx == text_idx
            ]

            chunk_indices = torch.tensor(
                chunk_indices,
                device=self.device,
                dtype=torch.long,
            )

            # Chunk weights
            chunk_mask = mask[
                chunk_indices
            ]

            chunk_lengths = (chunk_mask.sum(dim=(1, 2, 3)).to(hidden_states.dtype))
            weights = (chunk_lengths / chunk_lengths.sum())

            # Combine hidden states
            caption_hidden_states = (hidden_states[chunk_indices] * weights[:, None, None]).sum(dim=0)

            # Projected
            caption_projected = (projected[chunk_indices] * weights[:, None]).sum(dim=0)

            final_hidden_states.append(caption_hidden_states)
            final_projected.append(caption_projected)

        # Stack back into batch
        hidden_states = torch.stack(final_hidden_states, dim=0)
        projected = torch.stack(final_projected,dim=0)

        return hidden_states, projected


class VisionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda", random_init=False):
        super().__init__()
        self.device = device
        if random_init:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config).to(device)
        else:
            self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        if "mae" in model_name.lower():
            self.model.config.mask_ratio = 0.0

    def forward(
        self,
        images,
    ):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"]

        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = torch.stack(outputs["hidden_states"][1:]).permute(1, 0, 2, 3)
        hidden_states = hidden_states[:, :, 0, :]

        return hidden_states, None


class TextEncoder(nn.Module):
    def __init__(self, model_name="bigscience/bloomz-1b1", device="cuda", random_init=False):
        super().__init__()
        self.device = device
        if random_init:
            self.model = load_llm(model_name, from_init=True)
        else:
            self.model = load_llm(model_name)
        self.tokenizer = load_tokenizer(model_name)

    def forward(
        self,
        texts,
    ):  
        tokens = self.tokenizer(texts, padding="longest", return_tensors="pt").to(
            self.model.get_input_embeddings().weight.device
        )

        outputs = self.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )

        # average pooling
        feats = torch.stack(outputs["hidden_states"][1:]).permute(1, 0, 2, 3)
        mask = tokens["attention_mask"].unsqueeze(-1).unsqueeze(1)
        feats = (feats * mask).sum(2) / mask.sum(2)

        return feats, None


class NVEmbedEncoder(nn.Module):
    """
    NOTE: This module requires a specific environment to run NV-Embed-v2 correctly due to compatibility issues:
        sentence_transformers==2.7.0
        transformers==4.37.2
        torch==2.2.0+cu121
    https://huggingface.co/nvidia/NV-Embed-v2/discussions/45
    """

    def __init__(self, model_name="nvidia/NV-Embed-v2", device=None, random_init=False):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if random_init:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_config(config, trust_remote_code=True).to(device)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        mod = importlib.import_module(self.model.__class__.__module__)
        self.input_transform_func = getattr(mod, "input_transform_func")

    def forward(self, texts):

        batch_dict = self.input_transform_func(
            self.model.tokenizer,
            {"input_texts": [text for text in texts]},
            always_add_eos=True,
            max_length=1024,
            instruction="",
        )

        features = self.model.prepare_kwargs_from_batch(
            batch_dict, instruction_lens=0, device=self.model.device
        )

        outputs = self.model.embedding_model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

        embeds = self.model.latent_attention_model(
            outputs.last_hidden_state,
            features["pool_mask"],
        )
        
        hidden_states = outputs.hidden_states
        hidden_states = torch.stack(hidden_states[1:], dim=-1).permute(0, 3, 1, 2)

        # average pooling
        mask = features["attention_mask"].to(hidden_states.dtype).unsqueeze(-1).unsqueeze(1)
        hidden_states = (hidden_states * mask).sum(2) / mask.sum(2)
        # append the latent attention embedding to the pooled hidden states
        embeds = embeds.unsqueeze(1)
        combined = torch.cat([hidden_states, embeds], dim=1)

        return combined, None


class Qwen3VLEncoder(nn.Module):
    """
    Layer-wise Qwen3-VL-Embedding features extracted the way the model builds
    its embedding:
    chat template with system instruction + generation prompt, left padding,
    and the hidden state of the LAST token at every LLM layer.

    Handles both modalities with the same model:
      - images: hidden_states = pooled ViT block states  -> (B, N_vision, 1152)
                projected     = last-token LLM states     -> (B, N_llm, 4096)
      - texts:  hidden_states = last-token LLM states     -> (B, N_llm, 4096)
                projected     = None
    """

    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-Embedding-8B",
        device="cuda",
        torch_dtype=torch.bfloat16,
        instruction="Represent the user's input.",
        random_init=False,
    ):
        super().__init__()

        self.instruction = instruction

        if random_init:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_config(config, trust_remote_code=True).to(device=device, dtype=torch_dtype)
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        # last position must be a real token for last-token pooling
        self.processor.tokenizer.padding_side = "left"

    def _build_inputs(self, inputs, is_image):
        messages = []
        for x in inputs:
            content = {"type": "image", "image": x} if is_image else {"type": "text", "text": x}
            messages.append([
                {"role": "system", "content": [{"type": "text", "text": self.instruction}]},
                {"role": "user", "content": [content]},
            ])

        prompts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        processor_kwargs = {"text": prompts, "padding": True, "return_tensors": "pt"}
        if is_image:
            processor_kwargs["images"] = list(inputs)

        return self.processor(**processor_kwargs).to(self.model.device)

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        is_image = not isinstance(inputs[0], str)
        model_inputs = self._build_inputs(inputs, is_image)

        assert model_inputs["attention_mask"][:, -1].all(), "expected left padding"

        # record ViT block outputs (pre-merger) for image inputs
        raw_vision_blocks = []
        hooks = []
        if is_image:
            def hook(module, input, output):
                raw_vision_blocks.append(output[0] if isinstance(output, tuple) else output)

            for block in self.model.visual.blocks:
                hooks.append(block.register_forward_hook(hook))

        try:
            with torch.inference_mode():
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            for h in hooks:
                h.remove()

        # hidden_states[0] is the embedding output; the last entry is the
        # final-normed last_hidden_state, i.e. what the embedding pools
        llm_hidden_states = torch.stack(outputs.hidden_states[1:], dim=1)  # (B, N_llm, S, 4096)
        last_token = llm_hidden_states[:, :, -1, :].float()                 # (B, N_llm, 4096)

        if not is_image:
            return last_token, None

        # mean-pool patches per image at every ViT block
        patch_counts = model_inputs["image_grid_thw"].prod(dim=-1).tolist()
        pooled_vision_layers = []
        for h in raw_vision_blocks:
            image_features = []
            offset = 0
            for n_patches in patch_counts:
                image_features.append(h[offset : offset + n_patches].mean(dim=0))
                offset += n_patches
            pooled_vision_layers.append(torch.stack(image_features))

        vision_hidden = torch.stack(pooled_vision_layers, dim=1).float()   # (B, N_vision, 1152)

        return vision_hidden, last_token
