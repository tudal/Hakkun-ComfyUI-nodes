import random
import io
import re
from PIL import Image
from PIL import ImageOps
import numpy as np
import torch
from torchvision.ops import masks_to_boxes
import comfy.utils
import os

TEXT_TYPE = "STRING"
INT_TYPE = "INT"
IMAGE_TYPE = "IMAGE"
UND = 'undefined'


def get_random_line(text, seed):
    if isEmpty(text):
        return ""

    lines = text.splitlines()
    random.seed(seed)
    random_line = random.choice(lines)
    return random_line


class MultiTextMerge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": (TEXT_TYPE, {"default": ', ', "multiline": False}),
            },
            "optional": {
                "s1": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "s2": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "s3": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "s4": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "s5": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "s6": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    FUNCTION = "concatenate_strings"

    CATEGORY = "Hakkun"

    def concatenate_strings(self, s1='', s2='', s3='', s4='', s5='', s6='', delimiter="_"):
        delimiter = delimiter.replace("\\n", "\n")
        strings = [s1, s2, s3, s4, s5, s6]
        strings = [s for s in strings if isOk(s)]
        concatenated_string = delimiter.join(strings)
        return concatenated_string,


class RandomLine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (TEXT_TYPE, {"default": '', "multiline": True}),
                "seed": (INT_TYPE, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    FUNCTION = "get_random_line"

    CATEGORY = "Hakkun"

    def get_random_line(self, text, seed):
        return get_random_line(text, seed),


class RandomLine4:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": (TEXT_TYPE, {"default": '', "multiline": True}),
                "text2": (TEXT_TYPE, {"default": '', "multiline": True}),
                "text3": (TEXT_TYPE, {"default": '', "multiline": True}),
                "text4": (TEXT_TYPE, {"default": '', "multiline": True}),
                "seed": (INT_TYPE, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "delimiter": (TEXT_TYPE, {"default": ', ', "multiline": False}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    FUNCTION = "get_random_line4"

    CATEGORY = "Hakkun"

    def get_random_line(self, textt, seed):
        lines = textt.splitlines()
        random.seed(seed)
        random_line = random.choice(lines)
        return random_line

    def get_random_line4(self, seed, delimiter, text1='', text2='', text3='', text4=''):
        random.seed(seed)

        texts = []
        if isOk(text1):
            texts.append(get_random_line(text1, seed))
        if isOk(text2):
            texts.append(get_random_line(text2, seed))
        if isOk(text3):
            texts.append(get_random_line(text3, seed))
        if isOk(text4):
            texts.append(get_random_line(text4, seed))

        delimiter = delimiter.replace("\\n", "\n")

        text = delimiter.join(texts)

        return text,


class PromptParser:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (TEXT_TYPE, {"default": '', "multiline": True}),
                "tags_file": ("STRING", {"default": '', "multiline": False}),
                "seed": (INT_TYPE, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "extra1": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "extra2": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "tags": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE, TEXT_TYPE, TEXT_TYPE)
    RETURN_NAMES = ("positive", "negative", "debug")
    FUNCTION = "parse_prompt"

    CATEGORY = "Hakkun"

    def format_commas(self, input):
        formatted_string = input.replace(' ,', ',').replace(',', ', ')
        return formatted_string

    def line_perc(self, input):
        pattern = r'(\d+)%'
        match = re.search(pattern, input)
        if match:
            percentage = int(match.group(1)) / 100.0
            return percentage
        else:
            return 0.5

    def get_perc_text(self, input):
        pattern = r'(\d+)%(.*)'
        match = re.search(pattern, input)
        if match:
            rest_of_string = match.group(2)
            return rest_of_string.strip()
        else:
            return input

    def parse(self, text,  delimiter=","):
        result = []
        positives = []
        negatives = []

        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("^"):
                break
            if line.startswith("!"):
                continue
            if self.is_empty_or_whitespace(line):
                continue

            if line.startswith("?"):
                line = line[1:]
                perc = self.line_perc(line)
                if random.random() > perc:
                    continue
                line = self.get_perc_text(line)

            if "@" in line:
                positive, negative = line.split("@")
                positives.append(positive)
                negatives.append(negative)
            else:
                positives.append(line)

        positives = self.remove_empty(positives)
        negatives = self.remove_empty(negatives)

        if positives:
            result.append(delimiter.join(positives))
        else:
            result.append("")
        if negatives:
            result.append(delimiter.join(negatives))
        else:
            result.append("")

        return result

    def is_empty_or_whitespace(self, input):
        return input.strip() == ''

    def replace_multiple_spaces(self, input):
        return re.sub(r'\s+', ' ', input.strip())

    def replace_random(self, match):
        options = match.group(1).split('*')
        return random.choice(options)

    def select_random_from_braces(self, input):
        pattern = r'\[(.*?)\]'
        result = re.sub(pattern, self.replace_random, input)
        return result

    def randomly_select_string_with_weight(self, arr, n, seed):
        weights = []
        strings_without_weight = []
        weight_pattern = r'(\*(\d+)\*|\d+:)'

        for string in arr:
            match = re.search(weight_pattern, string)
            if match:
                weight_str = match.group(1)

                # Extract "*weight*" or "weight:" patterns
                if ':' in weight_str:
                    weight = int(weight_str[:-1])
                else:
                    weight = int(weight_str.strip('*'))
                weights.append(weight)
                strings_without_weight.append(
                    string.replace(match.group(), ''))
            else:
                # If no weight pattern is found, consider weight as 1
                weights.append(100)
                strings_without_weight.append(string)

        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]

        # Select no more than the number of strings available
        if n > len(strings_without_weight):
            n = len(strings_without_weight)

        # Use [!...] to force the use of seed inside bracket
        np_seed = seed % (2**32)
        rng = np.random.default_rng(np_seed)

        selected_string = rng.choice(
            strings_without_weight, n,  p=normalized_weights, replace=False)
        return selected_string

    # "I went there with [a [fast|slow] [car|[boat|yaht]]|an [expensive|cheap] [car|[boat|yaht]]]"
    # [[*10*pink|blue] bedroom*100*|city at [day|night] with [cars|trains|rockets]]
    # [*150*car|boat*30*|bi*80*ke]
    def select_random(self, text, seed):
        seed_container = [int(seed)]

        def random_choice(match):
            freeze_value, num_choices, options_str = match.groups()
            n = 1

            if num_choices:
                if '-' in num_choices:
                    start, end = map(int, num_choices.split('-'))
                    n = random.randint(start, end)
                else:
                    n = int(num_choices)

            options = options_str.split('|')

            # If [!...], freeze the value by always using the same seed
            arr_seed = seed if freeze_value else seed_container[0]
            seed_container[0] += 1

            return ' '.join(self.randomly_select_string_with_weight(options, n, arr_seed))

        pattern = r'\[(!)?(?:(\d+-\d+|\d*)#)?([^\[\]]+)\]'

        while re.search(pattern, text):
            text = re.sub(pattern, lambda match: random_choice(match), text)

        return text

    def remove_empty(self, arr):
        return [s for s in arr if s.strip()]

    def process_extra(self, text, placeholder, extra=None):
        if isEmpty(extra):
            if placeholder in text:
                return text.replace(placeholder, '')
            return text
        if placeholder in text:
            return text.replace(placeholder, extra)
        return extra + ', ' + text

    def fix_commas(self, text):
        elements = text.split(",")
        elements = [element.strip() for element in elements]
        elements = [element for element in elements if element]
        return ", ".join(elements)

    def parse_prompt(self, prompt, tags_file, seed, extra1=None, extra2=None, tags=None):
        random.seed(seed)
        if isOk(tags_file):
            tags = load_text(tags_file)

        prompt = remove_comments(prompt)

        prompt = self.process_extra(prompt, "<extra2>", extra2)
        prompt = self.process_extra(prompt, "<extra1>", extra1)

        if isOk(tags):
            tags = remove_empty_lines(tags)
            tags_dict = multiline_string_to_dict(tags)
            prompt = replace_placeholders(prompt, tags_dict)

        raw = prompt

        prompt = prompt.replace("em:", "embedding:")

        prompt = self.select_random(prompt, seed=seed)

        result = self.parse(prompt)

        result[0] = self.fix_commas(result[0])
        result[1] = self.fix_commas(result[1])

        return (
            result[0],
            result[1],
            'POSITIVE:\n' + result[0] +
            '\n\nNEGATIVE:\n' + result[1] +
            '\n\nseed:' + str(seed) +
            '\n\nextra1:' + (extra1 or "<none>") +
            '\nextra2:' + (extra2 or "<none>") +
            '\n\nRAW:\n' + raw
        )

# Tensor to PIL


def tensor2pil(image):
    # Assuming image is a 4D tensor with shape (1, 1, height, width)
    image = image.squeeze(0)  # Remove the batch dimension
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))


class CalculateUpscale:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IMAGE_TYPE,),
                "target_height": (INT_TYPE, {"default": 1920, "min": 0, "step": 1}),
                "tiles_in_x": (INT_TYPE, {"default": 1, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = (INT_TYPE, "FLOAT")
    RETURN_NAMES = ("tile_size", "upscale")
    FUNCTION = "calculate"

    CATEGORY = "Hakkun"

    def calculate(self, image, tiles_in_x, target_height):
        image_size = image.size()
        img_width = int(image_size[2])
        img_height = int(image_size[1])

        upscale = target_height/img_height

        upscaled_width = img_width * upscale

        tile_size = upscaled_width/tiles_in_x

        return tile_size, upscale


class ImageResizeToWidth:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IMAGE_TYPE,),
                "target_width": (INT_TYPE, {"default": 1920, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = (IMAGE_TYPE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "calculate"

    CATEGORY = "Hakkun"

    def calculate(self, image, target_width):
        image_size = image.size()
        img_width = int(image_size[2])
        scale_by = target_width/img_width
        return upscale(image, 'lanczos', scale_by)


class ImageResizeToHeight:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IMAGE_TYPE,),
                "target_height": (INT_TYPE, {"default": 1920, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = (IMAGE_TYPE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "calculate"

    CATEGORY = "Hakkun"

    def calculate(self, image, target_height):
        image_size = image.size()
        img_height = int(image_size[1])
        scale_by = target_height/img_height
        return upscale(image, 'lanczos', scale_by)


def upscale(image, upscale_method, scale_by):
    samples = image.movedim(-1, 1)
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    s = comfy.utils.common_upscale(
        samples, width, height, upscale_method, "disabled")
    s = s.movedim(1, -1)
    return (s,)


class ImageSizeToString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IMAGE_TYPE,),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    RETURN_NAMES = ("size",)
    FUNCTION = "calculate"

    CATEGORY = "Hakkun"

    def calculate(self, image):
        image_size = image.size()
        img_width = int(image_size[2])
        img_height = int(image_size[1])

        size = str(img_width)+'x'+str(img_height)

        return size,


class AnyConverter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "int_": ("INT", {"forceInput": True}),
                "float_": ("FLOAT", {"forceInput": True}),
                "number_": ("NUMBER", {"forceInput": True}),
                "string_": (TEXT_TYPE, {"forceInput": True}),
                "seed_": ("SEED", ),
                "str_": ("STR", ),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "NUMBER", TEXT_TYPE, "SEED", "STR")
    FUNCTION = "convert"
    CATEGORY = "Hakkun"

    def string_to_int(self, s):
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return 0

    def string_to_number(self, s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def convert(self, int_=None, float_=None, number_=None, string_=None, seed_=None, str_=None):
        if isOk(str_):
            string_ = str_
        if isOk(int_):
            value = int_
        elif isOk(float_):
            value = float_
        elif isOk(number_):
            value = number_
        elif isOk(string_):
            return (self.string_to_int(string_),
                    self.string_to_number(string_),
                    self.string_to_number(string_),
                    string_,
                    {"seed": self.string_to_int(string_), },
                    string_,)
        elif isOk(seed_):
            value = seed_.get('seed')
        else:
            value = 0
        return int(value), float(value), float(value), str(value), {"seed": int(value), }


def multiline_string_to_dict(input_string):
    lines = input_string.strip().split('\n')
    result_dict = {}
    current_key = None
    current_value = ""

    for line in lines:
        if line.startswith('>>>'):
            # If a new key is encountered, save the previous key and value
            if current_key is not None:
                result_dict[current_key] = current_value
            current_key = line[3:]
            current_value = ""
        elif current_value == "":
            current_value = line.strip()
        else:
            current_value += ', ' + line.strip()

    # for key, value in result_dict.items():
        # result_dict[key] = remove_trailing_newline(value)

    # Add the last key and value to the dictionary
    if current_key is not None:
        result_dict[current_key] = current_value

    return result_dict


def replace_placeholders(input_string, dictionary):
    for key, value in dictionary.items():
        placeholder = f'<{key}>'
        input_string = input_string.replace(placeholder, value)
    return input_string


def remove_empty_lines(input_string):
    lines = input_string.split('\n')  # Split the input string into lines
    # Filter out lines with only whitespace
    non_empty_lines = [line for line in lines if line.strip()]
    # Join the non-empty lines back into a string
    return '\n'.join(non_empty_lines)


def remove_trailing_newline(input_string):
    while input_string.endswith('\n'):
        input_string = input_string[:-1]
    return input_string


class LoadRandomImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": (TEXT_TYPE, {"default": ""}),
                "subdirectories": (["ignore", "include"],),
                "seed": (INT_TYPE, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = (IMAGE_TYPE, TEXT_TYPE)
    RETURN_NAMES = ("image", "file name")
    FUNCTION = "load_images"

    CATEGORY = "Hakkun"

    def load_images(self, directory: str, subdirectories: str, seed):
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Directory '{directory} cannot be found.'")

        # Function to recursively find files in subdirectories
        def find_files_in_dir(directory, subdirectories):
            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            if subdirectories == "include":
                dir_files = []
                for root, dirs, files in os.walk(directory):
                    dir_files += [os.path.join(root, f) for f in files if any(
                        f.lower().endswith(ext) for ext in valid_extensions)]
            else:
                dir_files = [os.path.join(directory, f) for f in os.listdir(
                    directory) if any(f.lower().endswith(ext) for ext in valid_extensions)]
            return dir_files

        dir_files = find_files_in_dir(directory, subdirectories)

        if not dir_files:
            raise FileNotFoundError(
                f"No valid image files in directory '{directory}'.")

        dir_files = sorted(dir_files)

        random.seed(seed)
        random_index = random.randint(0, len(dir_files) - 1)
        image_path = dir_files[random_index]

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        file_name = get_file_name_without_extension(image_path)

        return (image, file_name)


class LoadText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    FUNCTION = "load_file"

    CATEGORY = "Hakkun"

    def load_file(self, file_path=''):
        return (load_text(file_path),)


def load_text(file_path=''):
    with open(file_path, 'r', encoding="utf-8", newline='\n') as file:
        text = file.read()
    lines = []
    for line in io.StringIO(text):
        if not line.strip().startswith('#'):
            if (not line.strip().startswith("\n")
                    or not line.strip().startswith("\r")
                    or not line.strip().startswith("\r\n")):
                line = line.replace("\n", '').replace(
                    "\r", '').replace("\r\n", '')
            lines.append(line.replace("\n", '').replace(
                "\r", '').replace("\r\n", ''))
    return "\n".join(lines)


def remove_comments(string):
    pattern = r"/\*.*?\*/"
    modified_string = re.sub(pattern, "", string)
    return modified_string


def get_file_name_without_extension(file_path):
    file_name_with_extension = os.path.basename(file_path)
    file_name, _ = os.path.splitext(file_name_with_extension)
    return file_name


def isEmpty(value):
    return value is None or (isinstance(value, str) and value == "")


def isOk(value):
    return not isEmpty(value)


NODE_CLASS_MAPPINGS = {
    "Multi Text Merge": MultiTextMerge,
    "Random Line": RandomLine,
    "Random Line 4": RandomLine4,
    "Prompt Parser": PromptParser,
    "Calculate Upscale": CalculateUpscale,
    "Image size to string": ImageSizeToString,
    "Any Converter": AnyConverter,
    "Image Resize To Height": ImageResizeToHeight,
    "Image Resize To Width": ImageResizeToWidth,
    "Load Random Image": LoadRandomImage,
    "Load Text": LoadText,
}
