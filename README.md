# Hakkun-ComfyUI-nodes
ComfyUI extra nodes. Mostly prompt parsing

# Installation
Just put `hakkun_nodes.py` into `ComfyUI\custom_nodes`

Drag and drop ```prompt_parser_example_workflow.png``` into ComfyUI to check use example for all nodes.

Other custom nodes used in example:

quality of life

https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92

ComfyUI-Custom-Scripts

https://github.com/pythongosssss/ComfyUI-Custom-Scripts

# Custom nodes

All nodes can be found in "Hakkun" category.

## Prompt parser

![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/c31f7513-ea33-4537-a32a-ca2a74f76804)

Allows you to write whole positive and negative prompt in one text field.

You can write on multiple lines. All lines will be joined by "," and cleaned of unnecessary white space.

### Special syntax examples:

Everything after ```@``` will be used as negative prompt.
```
3d render @ drawing, painting
```
positive: ```3d render```

negative: ```drawing, painting```

All lines starting with ```!``` will be ignored.

All lines starting with `?` will be randomly ignored with a default 50% chance.

You can modify the default 50% chance by specifying different percentages after `?`.

```
?20% explosion
```
This will give you a 20% chance to include the word 'explosion'.


To randomly select from a list, use this format:
```
[dog|cat|horse]
```

You can nest these lists on multiple levels:
```
[town|city] full of [[fast|slow] [cars|[bicycles|motorbikes]]|[[loud|cute] [horses|[ducks|chickens]]]]
```
This will generate sentences like:
```
town full of cute horses
city full of fast bicycles
```

To adjust the chances of selecting each element separately, assign weights using the following format:
```
[*150*car|*30*boat|train|*80*bike]
```
If a weight is not specified for an element, the default weight of 100 will be used. Weights can be placed anywhere within the elements.

This also works with nested elements:
```
[[*10*pink|blue] bedroom*30*|*120*city at [day|*150*night] with [cars|trains|*10*rockets]]
```
In this example, a pink bedroom will be very rare.

There's also the option to insert external text in ```<extra1>``` or ```<extra2>``` placeholders.

Include ```<extra1>``` and/or ```<extra2>``` anywhere in the prompt, and the provided text will be inserted before parsing. If you don't specify these triggers but provide text, it will be pasted at the beginning by default (as positive).

All occurrences of ```em:``` will be replaced with ```embedding:```.

If a line starts with ```^```, everything after that symbol and the **next lines** will be ignored.

You can set the ```seed``` as an input to control randomness along with the rest of the workflow.

Using the ```debug``` output will provide you with all the information about the generated prompt.


## Multi Text Merge
Allows to join multiple texts by specified delimiter. Put ```\n``` to use new line as delimiter.
No need to keep any order. Empty inputs will be ignored.

## Random Line
Will output random line

## Random Line 4
Will output random lines from 4 textfields/inputs and join them by specified delimiter. Put ```\n``` to use new line as delimiter.

## Calculate Upscale
Outputs target upscale float value based on input image height.
Also calculates tile size (with and height) for tools like UltimateSDUpscale by specified number of horizontal tiles.

## Image size to string
Outputs input image size in format: ```512x768```

## Any Converter
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/c3281a50-8873-4dd5-8f01-8ba347c0874c)

