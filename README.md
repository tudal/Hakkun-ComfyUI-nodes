# Hakkun-ComfyUI-nodes
ComfyUI extra nodes. Mostly prompt parsing

# Installation
Just put hakkun_nodes.py into ComfyUI\custom_nodes

Drag and drop ```prompt_parser_example_workflow.png``` into ComfyUI to check use example for all nodes.

Used other custom nodes in example:

quality of life

https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92

ComfyUI-Custom-Scripts

https://github.com/pythongosssss/ComfyUI-Custom-Scripts

# Custom nodes

All nodes can be found in "Hakkun" category.

## Prompt parser

![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/3ff89cc3-bafd-4c2e-ac0c-f4afefc03d6a)

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

All lines starting with ```?``` will be randomly ignored with 50% chance.

You can modify default 50% chance by adding percentages after ```?```.
```
?20% explosion
```
Will give you 20% chances to include ```explosion```

To select random from list you can use this format:
```
[dog|cat|horse]
```
You can nest those on multiple levels:
```
[town|city] full of [[fast|slow] [cars|[bicycles|motorbikes]]|[[loud|cute] [horses|[ducks|chickens]]]]
```
Will get you sentences like:
```
town full of cute horses
city full of fast bicycles
```

To adjust the chances of selecting each element separately, you can assign weights by using the following format:
```
[*150*car|boat*30*|train|bi*80*ke]
```
```train``` dont have weight specified so default 100 will be used.
Weight can be put anywhere.


This works also with nested elements:
```
[[*10*pink|blue] bedroom*30*|*200*city at [day|night] with [cars|trains|rockets]]
```
In this example pink bedroom will be very rare.

There's also option to insert external text in ```extra1```, ```extra1``` options.
Put ```<extra1>``` and/or ```<extra2>``` anywhere in prompt and it will be pasted before partsing. If you dont specify those triggers but provide text then it will be pasted at beginning (as positive).

All ```em:``` will be replaced to ```embedding:```

If line is starting with ```^``` everything after that and **next lines** be ignored.

Set ```seed``` as input to control randomness together with rest of workflow.

```debug``` output will give you all information about generated prompt

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
