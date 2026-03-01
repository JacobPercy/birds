# Sketch2Bird

Sketch2Bird generates realistic bird images from simple line sketches using a pix2pix-style U-Net generator.

## How it works

1. Sketch input (manual drawing or edge-based sketch)
2. Resize to 256x256
3. Convert to RGB
4. Normalize to [-1, 1]
5. Forward pass through the generator
6. Convert output back to an RGB image

## Model

- Conditional image-to-image translation
- U-Net generator with encoder-decoder structure
- Skip connections
- Tanh output
- PyTorch

## Results

### Sample panels

![row_subset_5pairs_01](media/row_subset_5pairs_01.png)
![row_subset_5pairs_02](media/row_subset_5pairs_02.png)
![row_subset_5pairs_03](media/row_subset_5pairs_03.png)
![row_subset_5pairs_04](media/row_subset_5pairs_04.png)
![row_subset_5pairs_05](media/row_subset_5pairs_05.png)
![row_subset_5pairs_06](media/row_subset_5pairs_06.png)
![row_subset_5pairs_07](media/row_subset_5pairs_07.png)

### Animations

![flip_sketch_generated_all](media/flip_sketch_generated_all.gif)
![mosaic_mode_1](media/mosaic_mode_1.gif)
![strip_reveal_story](media/strip_reveal_story.gif)

## Drawing interface

Screenshots:

![ss1](media/ss1.png)
![ss2](media/ss2.png)
![ss3](media/ss3.png)
