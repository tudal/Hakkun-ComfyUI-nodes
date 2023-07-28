import random
import re
from PIL import Image
import numpy as np

TEXT_TYPE = "STRING"
INT_TYPE = "INT"

def get_random_line(text, seed):
    if text is None or text.strip() == '':
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
        strings = [s for s in [s1, s2, s3, s4, s5, s6] if s]
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
        # Split the multiline string into lines
        lines = textt.splitlines()

        random.seed(seed)

        # Choose a random line from the list of lines
        random_line = random.choice(lines)

        return random_line

    def get_random_line4(self, seed, delimiter, text1='', text2='', text3='', text4=''):
        random.seed(seed)

        texts = []
        if len(text1) > 0: texts.append(get_random_line(text1,seed))
        if len(text2) > 0: texts.append(get_random_line(text2,seed))
        if len(text3) > 0: texts.append(get_random_line(text3,seed))
        if len(text4) > 0: texts.append(get_random_line(text4,seed))

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
                "seed": (INT_TYPE, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "extra1": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "extra2": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,TEXT_TYPE,TEXT_TYPE)
    RETURN_NAMES = ("positive","negative","debug")
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

    # ["*150*car", "boat*30*", "*80*bike"]
    def randomly_select_string_with_weight(self, arr):
        weights = []
        strings_without_weight = []
        weight_pattern = r'\*(\d+)\*'

        for string in arr:
            match = re.search(weight_pattern, string)
            if match:
                weight = int(match.group(1))
                weights.append(weight)
                strings_without_weight.append(string.replace(match.group(), ''))
            else:
                # If no weight pattern is found, consider weight as 1
                weights.append(100)
                strings_without_weight.append(string)

        selected_string = random.choices(strings_without_weight, weights=weights)[0]
        return selected_string

    # "I went there with [a [fast|slow] [car|[boat|yaht]]|an [expensive|cheap] [car|[boat|yaht]]]"
    # [[*10*pink|blue] bedroom*100*|city at [day|night] with [cars|trains|rockets]]
    # [*150*car|boat*30*|bi*80*ke]
    def select_random(self, text):
        def random_choice(match):
            options = match.group(1).split('|')
            return self.randomly_select_string_with_weight(options)

        pattern = r'\[([^\[\]]+)\]'

        while re.search(pattern, text):
            text = re.sub(pattern, random_choice, text)

        return text

    def remove_empty(self, arr):
        return [s for s in arr if s.strip()]

    def process_extra(self, text, placeholder, extra=None):
        if extra is None:
            return text
        if placeholder in text:
            return text.replace(placeholder, extra)
        return extra +', '+ text

    def fix_commas(self, text):
        elements = text.split(",")
        elements = [element.strip() for element in elements]
        elements = [element for element in elements if element]
        return ", ".join(elements)

    def parse_prompt(self, prompt, seed, extra1=None, extra2=None):
        random.seed(seed)

        prompt = self.process_extra(prompt, "<extra2>", extra2)
        prompt = self.process_extra(prompt, "<extra1>", extra1)

        raw = prompt

        prompt = prompt.replace("em:", "embedding:")

        prompt = self.select_random(prompt)

        result = self.parse(prompt)

        result[0] = self.fix_commas(result[0])
        result[1] = self.fix_commas(result[1])

        return  (
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
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class CalculateUpscale:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_height": (INT_TYPE, {"default": 1920, "min": 0, "step": 1}),
                "tiles_in_x": (INT_TYPE, {"default": 1, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = (INT_TYPE, "FLOAT")
    RETURN_NAMES = ("tile_size","upscale")
    FUNCTION = "calculate"

    CATEGORY = "Hakkun"

    def calculate(self, image, tiles_in_x, target_height):
        image = tensor2pil(image)
        img_width = image.size[0]
        img_height = image.size[1]

        upscale = target_height/img_height

        upscaled_width = img_width * upscale

        tile_size=upscaled_width/tiles_in_x

        return tile_size, upscale

class ImageSizeToString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    RETURN_NAMES = ("size",)
    FUNCTION = "calculate"

    CATEGORY = "Hakkun"

    def calculate(self, image):
        image = tensor2pil(image)
        img_width = image.size[0]
        img_height = image.size[1]

        size = str(img_width)+'x'+str(img_height)

        return size,

class ShowImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True

    CATEGORY = "Hakkun"

    def notify(self, image):
        # imagee = tensor2pil(image[0])
        # img_width = imagee.size[0]
        # img_height = imagee.size[1]
        text = 'YOOOOOOOOOOOOOOOOOOOOO'
        return {"ui": {"text": text}, "result": text}

NODE_CLASS_MAPPINGS = {
    "Multi Text Merge": MultiTextMerge,
    "Random Line": RandomLine,
    "Random Line 4": RandomLine4,
    "Prompt Parser": PromptParser,
    "Calculate Upscale": CalculateUpscale,
    "Image size to string": ImageSizeToString,
}


# Test cases
# print(test("[45%]purple")) # 0.45
# print(test("[75%]orange")) # 0.75
# print(test("[100%]blue"))  # 1.0
# print(test("yellow"))      # None
# print(test("[30%]"))      # 0.3

# def test(input):
#     print(select_random(input))
#
# test("I like [apple|banana|cherry]");
# test("[Hello world|Hi|Hey] there, [how are you?|what's up?]");
# test("I went there with [a [fast|slow] [car|boat]|an [expensive|cheap] [car|boat]]");
# test("This is a [test]");
# test("The [quick|lazy] [dog|fox] jumps [over|under] the [brown|black] [fence|wall].");
# test("Empty options: [].");

