import os
import torch
import folder_paths
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.v2 import ToPILImage
import cv2  # pip install opencv-python
from PIL import Image


class MiniCPM_VQA_Vts:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "split_text": ("STRING", {"default": "-----", "multiline": False}),
                "character_face_text": ("STRING", {"default": "", "multiline": True}),
                "character_body_text": ("STRING", {"default": "", "multiline": True}),
                "character_muscle_text": ("STRING", {"default": "", "multiline": True}),
                "character_face_comma_text": ("STRING", {"default": "", "multiline": True}),
                "character_comma_text": ("STRING", {"default": "", "multiline": True}),
                "character_body_tags_text": ("STRING", {"default": "", "multiline": True}),
                "character_ethnicity_tags_text": ("STRING", {"default": "", "multiline": True}),
                "environment_text": ("STRING", {"default": "", "multiline": True}),
                "environment_comma_text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    ["MiniCPM-V-2_6-int4", "MiniCPM-Llama3-V-2_5-int4", "Vision-8B-MiniCPM-2_5-Uncensored-and-Detailed-4bit"],
                    {"default": "MiniCPM-V-2_6-int4"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.8,
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 100,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.05,
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 2048,
                    },
                ),
                "video_max_num_frames": (
                    "INT",
                    {
                        "default": 64,
                    },
                ),  # if cuda OOM set a smaller number
                "video_max_slice_nums": (
                    "INT",
                    {
                        "default": 2,
                    },
                ),  # use 1 if cuda OOM and video resolution >  448*448
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
            },
            "optional": {
                "source_video_path": ("PATH",),
                "source_image_path": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING"
    )

    RETURN_NAMES = (
        "character_face_text",
        "character_body_text",
        "character_muscle_text",
        "character_face_comma_text",
        "character_comma_text",
        "character_body_tags_text",
        "character_ethnicity_tags_text",
        "environment_text",
        "environment_comma_text",
        "character_neg_body_tags_text",
        "character_neg_ethnicity_tags_text",
    )
    FUNCTION = "inference"
    CATEGORY = "Comfyui_MiniCPM-V-2_6-int4"

    def encode_video(self, source_video_path, MAX_NUM_FRAMES):
        def uniform_sample(l, n):  # noqa: E741
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        cap = cv2.VideoCapture(source_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames:", total_frames)
        avg_fps = cap.get(cv2.CAP_PROP_FPS)
        print("Get average FPS(frame per second):", avg_fps)
        sample_fps = round(avg_fps / 1)  # FPS
        duration = total_frames / avg_fps
        print("Total duration:", duration, "seconds")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video resolution(width x height):", width, "x", height)

        frame_idx = [i for i in range(0, total_frames, sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = []
        for idx in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        print("num frames:", len(frames))
        return frames

    def calculate_results(self, text, split_text, images, video_max_slice_nums, top_p, top_k, temperature, repetition_penalty, max_new_tokens):
        # if text is empty, or whitespace, return empty string
        if not text or not text.strip():
            return ""
        
        # split text by newline
        texts = text.split(split_text)
        finalResults = []
        for text in texts:
            # remove any leading or trailing whitespace
            text = text.strip()
            #remove any leading or trailing newline characters
            text = text.strip("\n")
            # remove any leading or trailing whitespace
            text = text.strip()
            if images:
                msgs = [{"role": "user", "content": images + [text]}]
            else:
                msgs = [{"role": "user", "content": [text]}]
                # raise ValueError("Either image or video must be provided")

            params = {"use_image_id": False, "max_slice_nums": video_max_slice_nums}

            # offload model to CPU
            # self.model = self.model.to(torch.device("cpu"))
            # self.model.eval()

            result = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                **params,
            )
            print(result)
            finalResults.append(result)
        return finalResults


    def filter_values(self, input_string):
        try:
            # Check if the input is a list of strings
            if isinstance(input_string, list):
                # Join the list into a single string with comma and space
                input_string = ', '.join(input_string)

            # Remove any ` or ' characters
            input_string = input_string.replace("`", "").replace("'", "").replace("\r", "").replace("\n", "").strip()
            
            # Split the input string by commas
            value_pairs = input_string.split(', ')
            
            # Initialize lists to store the filtered value pairs
            filtered_values = []
            zero_or_less_values = []
            
            # Iterate through each value pair
            for pair in value_pairs:
                # Extract the key and numerical value from the pair
                key, value = pair.split(':')
                value = float(value.strip(')'))
                
                # Check if the key ends with " arms"
                if key.strip().endswith(" arms"):
                    # Reduce the numerical value
                    value /= 4

                # Check if the key ends with "chinese", "japanese", or "korean"
                if key.strip().endswith("chinese") or key.strip().endswith("japanese") or key.strip().endswith("korean"):
                    # Reduce the numerical value
                    value -= 0.2
                
                # Check if the key ends with "abs"
                if key.strip().endswith("abs"):
                    # Increase the numerical value
                    value *= 1.25

                # Round the value to 2 decimal places
                value = round(value, 2)

                # Check if the value is above 1.5
                if value > 1.5:
                    value = 1.5
                
                # Check if the numerical value is greater than zero
                if value > 0:
                    # If it is, add the pair to the filtered values list
                    filtered_values.append(f"{key}:{value})")
                else:
                    # Otherwise, add the pair to the zero or less values list
                    zero_or_less_values.append(f"{key}:1.2)")
            
            # Join the filtered value pairs back into a comma-separated string
            filtered_string = ', '.join(filtered_values)
            filtered_negative_string = ', '.join(zero_or_less_values)
            return filtered_string, filtered_negative_string
        except Exception as e:
            # Return the original input and an empty string in case of an error
            return input_string, ""

    def inference(
        self,
        split_text,
        character_face_text,
        character_body_text,
        character_muscle_text,
        character_face_comma_text,
        character_comma_text,
        character_body_tags_text,
        character_ethnicity_tags_text,
        environment_text,
        environment_comma_text,
        model,
        keep_model_loaded,
        top_p,
        top_k,
        temperature,
        repetition_penalty,
        max_new_tokens,
        video_max_num_frames,
        video_max_slice_nums,
        seed,
        source_image_path=None,
        source_video_path=None,
    ):
        if seed != -1:
            torch.manual_seed(seed)
        if model == "Vision-8B-MiniCPM-2_5-Uncensored-and-Detailed-4bit":
            model_id = "sdasd112132/Vision-8B-MiniCPM-2_5-Uncensored-and-Detailed-4bit"
        else:
            model_id = f"openbmb/{model}"

        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "prompt_generator", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_checkpoint,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_checkpoint,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
            )

        with torch.no_grad():
            if source_video_path:
                print("source_video_path:", source_video_path)
                images = self.encode_video(source_video_path, video_max_num_frames)
            elif source_image_path is not None:
                images = source_image_path.permute([0, 3, 1, 2])
                images = [ToPILImage()(img).convert("RGB") for img in images]
            else:
                images = None

            character_face_text_results = self.calculate_results(character_face_text, split_text, images, video_max_slice_nums, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
            character_body_text_results = self.calculate_results(character_body_text, split_text, images, video_max_slice_nums, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
            character_muscle_text_results = self.calculate_results(character_muscle_text, split_text, images, video_max_slice_nums, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
            # character_text_results is the character_body_text_results array concatenated to the character_face_text_results array
            character_text_results = character_face_text_results + character_body_text_results
            character_body_muscle_results = character_body_text_results + character_muscle_text_results
            character_text_muscle_results = character_face_text_results + character_body_text_results + character_muscle_text_results
            # character_text is the character_text_results array concatenated to a single string with a newline character as the separator and enclosed in ``` characters
            character_text = "```\n" + "\n".join(character_text_results) + "\n```"
            character_full_text = "```\n" + "\n".join(character_text_muscle_results) + "\n```"
            character_face_text = "```\n" + "\n".join(character_face_text_results) + "\n```"
            character_body_text = "```\n" + "\n".join(character_body_text_results) + "\n```"
            character_body_muscle_text = "```\n" + "\n".join(character_body_muscle_results) + "\n```"

            used_character_face_comma_text = character_comma_text + character_face_text
            character_face_comma_text_results = self.calculate_results(used_character_face_comma_text, split_text, None, video_max_slice_nums, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

            used_character_comma_text = character_face_comma_text + character_text
            character_comma_text_results = self.calculate_results(used_character_comma_text, split_text, None, video_max_slice_nums, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

            used_ethnicity_text = character_ethnicity_tags_text + character_full_text
            character_ethnicity_tags_text_results, character_ethnicity_tags_text_neg_results = self.filter_values(self.calculate_results(used_ethnicity_text, split_text, images, video_max_slice_nums, top_p, top_k, temperature, repetition_penalty, max_new_tokens))

            used_body_tags_text = character_body_tags_text + character_body_muscle_text
            character_body_tags_text_results, character_body_tags_text_neg_results = self.filter_values(self.calculate_results(used_body_tags_text, split_text, images, video_max_slice_nums, top_p, top_k, temperature, repetition_penalty, max_new_tokens))

            environment_text_results = self.calculate_results(environment_text, split_text, images, video_max_slice_nums, top_p, top_k, temperature, repetition_penalty, max_new_tokens)
            environment_text = "```\n" + "\n".join(environment_text_results) + "\n```"

            used_environment_comma_text = environment_comma_text + environment_text
            environment_comma_text_results = self.calculate_results(used_environment_comma_text, split_text, None, video_max_slice_nums, top_p, top_k, temperature, repetition_penalty, max_new_tokens)

            # offload model to GPU
            # self.model = self.model.to(torch.device("cpu"))
            # self.model.eval()
            if not keep_model_loaded:
                del self.tokenizer  # release tokenizer memory
                del self.model  # release model memory
                self.tokenizer = None  # set tokenizer to None
                self.model = None  # set model to None
                torch.cuda.empty_cache()  # release GPU memory
                torch.cuda.ipc_collect()

            return (
                character_face_text_results,
                character_body_text_results,
                character_muscle_text_results,
                character_face_comma_text_results,
                character_comma_text_results,
                character_body_tags_text_results,
                character_ethnicity_tags_text_results,
                environment_text_results,
                environment_comma_text_results,
                character_body_tags_text_neg_results,
                character_ethnicity_tags_text_neg_results
            )