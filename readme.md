# Movie Diffusion
By: Anton Forsman, Carlos Marí, Tim Olsen

Implementation of a diffusion model to generate movie posters.

Model checkpoint can be found at https://huggingface.co/anforsm/movie-diffusion/

## Sample outputs

### CIFAR-10 Class conditional sampling
In a 3x3 collage, each row and column is a different class. Same across collages.

![CIFAR-10 Class conditional sampling](./samples/cifar1.png)
![CIFAR-10 Class conditional sampling](./samples/cifar2.png)
![CIFAR-10 Class conditional sampling](./samples/cifar3.png)

### Movie posters 
These are 2x2 collages of movie posters.

![Conditional movie posters](./samples/cond1.png)
![Conditional movie posters](./samples/Unconditional3.png)
![Conditional movie posters](./samples/cond3.png)

### Inpainting
These are posters which have been inpainted in different ways.

![Inpainting](./samples/inpainting1.png)
![Inpainting](./samples/inpainting2.png)
![Inpainting](./samples/inpainting3.png)
