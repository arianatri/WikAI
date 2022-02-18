<h1><center> Image processing Reference </center></h1>
<hr/>

## Table of Contents

- [Image processing frameworks](#Image-processing-frameworks)
- [Image formats](#Image-formats)
- [Color spaces](#Color-spaces)
- [Blend Modes](#Blend-Modes)
- [Image operations](#Image-operations)
  * [Basic operations](#Basic-operations)
  * [Image arithmetic operation](#Image-arithmetic-operation)
  * [Geometric Operations](#Geometric-Operations)
  * [Thresholding](#Thresholding)
  * [Edge detection](#Edge-detection)
  * [Smoothing](#Smoothing)
  * [Sharpening](#Sharpening)
  * [Morphology](#Morphology)
- [Material](#Material)
  * [Books](#Books)
  * [Papers](#Papers)
  * [Blogs](#Blogs)
  
  
## Image processing frameworks

| Framework                                                    | Image | Video | C/C++ | Python |
| ------------------------------------------------------------ | ----- | ----- | ----- | ------ |
| [<img src="https://opencv.org/wp-content/uploads/2020/07/cropped-Fav-32x32.png" width=32/>OpenCV](https://opencv.org/) | :heavy_check_mark:     | :heavy_check_mark:     | :heavy_check_mark:     | :heavy_check_mark:      |
| [<img src="https://pillow.readthedocs.io/en/stable/_static/favicon.ico" width=32 />Pillow](https://pillow.readthedocs.io/en/stable/) | :heavy_check_mark:     |       | :heavy_check_mark:     |        |
| [<img src="https://scikit-image.org/_static/favicon.ico" width=32/>Scikit-image](https://scikit-image.org/) | :heavy_check_mark:     |       | :heavy_check_mark:     |        |

## Image formats

### Raster Image formats

<img src="https://upload.wikimedia.org/wikipedia/commons/8/83/Raster.png" width="50%" alt="Raster image example" />

| Format | Name                             | Extension    | Alpha              | Lossy             |  Bits per channel | Compression                |
|--------|----------------------------------|--------------|--------------------|-------------------|-------------------|----------------------------|
| **BMP** | Windows Bitmap                   | .bmp         |                    | :heavy_check_mark:| 8                 | RLE                        |
| **JPG** | Joint Photographic Experts Group | .jpeg, .jpg  |                    | :heavy_check_mark:| 8                 | DCT                        |
| **GIF** | GIF                              | .gif         | :heavy_check_mark: | :heavy_check_mark:| 8 (indexed)       |                            |
| **PNG** | Portable Network Graphics        | .png         | :heavy_check_mark: |                   | 1,2,4,8,16 (GRAY), 8, 16 (RGB)   | DEFLATE     |
| **TIFF** | Tagged Image File Format         | .tiff        | :heavy_check_mark: |                   | 8, 16             | PackBits, LZW, DCT, Huffman|

#### Vector Image formats

<img src="https://upload.wikimedia.org/wikipedia/commons/6/66/Bezier_curve.png" alt="Bezier curve example" width="50%" />

| Format  | Name                     | Extension   |
| ------- | ------------------------ | ----------- |
| **SVG** | Scalable Vector Graphics | .svg, .svgz |
| **EPS** | Encapsulated PostScript  | .eps        |

### Resources

1. [Full list of file formats](https://en.wikipedia.org/wiki/List_of_file_formats#Graphics)

## Color spaces

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/HSV_color_solid_cone_chroma_gray.png/1200px-HSV_color_solid_cone_chroma_gray.png?20100323051129" alt="Bezier curve example" width="50%" />

| Name         | Channels                                |
| ------------ | --------------------------------------- |
| **L** (Gray) | L=Luminance                             |
| **RGB**      | R=Red, G=Green, B=Blue                  |
| **RGBA**     | R=Red, G=Green, B=Blue, A=Alpha         |
| **CMYK**     | C=Cyan, M=Magenta, Y=Yellow, K=Black    |
| **HSL**      | H=Hue, S=Saturation, L=Luminance        |
| **HSV**      | H=Hue, S=Saturation, V=Value            |
| **YUV**      | Y=Yellow, U, V                          |
| **Lab**      | L=Luminance, a=Red-Green, b=Yellow-Blue |

## Image operations

### Basic operations

> Basic transformation that can be applied to image usually as preprocessing step

| Name                 | Description | Parameters  |
|----------------------|-------------|-------------|
| **Cropping**         | Crop a region of interest from an image | Crop location (x1,y1), (x2,y2) in some image coordinates |
| **Padding**          | Create a border around an image | pl=Left amount, pr=Right amount, pt=Top amount, pb=Bottom ammount |
| **Horizontal Flip**  | Flips the image horizontally ||
| **Vertical Flip**    | Flips the image vertically ||
| **Changing colorspaces** | Convert an image from one colorspace to another colorspace (e.g. RGB->HSL) | Source and target colorspace |

### Image arithmetic operation

<img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Blend_Modes_Darken%26Lighten.png?20201028004021" width="50%" alt="Blending modes example" />

> A set of functions to combine one or multiple images or binary masks into another one  

| Name                     | Description | Parameters  |
|--------------------------|-------------|-------------|
| **Image arithmetic**     | Merge two images by using operations like addition, substraction, multiplication |  |
| **Image Bitwise operations** | Binary operations to work with masks like AND (mask intersection), OR (mask addition), XOR (union-intersection) | |
| **Image blending**       | Merge two images using a threshold | alpha=threshold and a [blending mode](#Blend-modes) |

## Blend modes

| Type | Name | Effect | Equation |
|------|----------|------|--------|
| Normal |**Normal** | Returns the top image | f(a,b)=b |
| Darkening | **Multiply** | Darkens the parts of _b_ where _a_ is dark | f(a,b)=ab |
| Lightning | **Screen** | Lighten the parts of _b_ where _a_ is light |  f(a,b)=1-(1-a)(1-b) |
| - | **Overlay** | Darkens the parts of _b_ where _a_ is dark while lighten the parts of b where _a_ is light| f(a,b)= (b < 0.5) ? 2ba : 1-2(1-b)(1-a) |
| - | **Hard Light** |Same as overlay but with the order inverted | f(a,b)= (a < 0.5) ? 2ab : (1-2(1-a)(1-b)) |
| - | **Soft Light** | Similar to overlay but when b is pure black or white the result is not pure black or white | f(a,b) = (b < 0.5) ? (2ab + a² (1-2b)) : 2a(1-b) + √a (2b-1) |
| Darkening | **Color burn** |||

### Geometric Operations

<img src="https://live.staticflickr.com/2861/9279197332_ca6beb3760_b.jpg" alt="Perspective wrap example" width="50%" />

| Name                  | Description | Parameters  |
|-----------------------|-------------|-------------|
| **Resize**            | Scale (shrink or growing) the resolution of an image by applying some interpolation method | w=Scaling in width, h=scaling in height |
| **Translate**         | Shifting of an object's location while mainting the image dimension | x=Shift in left-right direction, y=shift in top-bottom direction |
| **Rotation**          | Rotation of an image by an angle |  θ rotation angle usually in degrees |
| **Affine Transformation** | A more generic linear transformation such that all parallel lines in the original image will still be parallel in the output image | Transformation matrix |
| **Perspective wrap**  | Change the perspective of an image | 4 2-d coords from the source location and 4 2-d coords correspoinding to the target location |

### Thresholding

<img src="https://upload.wikimedia.org/wikipedia/commons/7/7b/Thresholding.png" alt="Simple thresholding example" width="50%" />

| Name                | Description | Parameters  |
|---------------------|-------------|-------------|
| **Simple thresholding** | Select pixel with values larger than threhsold | L=lower threshold, U=upper threshold |
| **Otsu's thresholding** | Otsu's method determines an optimal global threshold value from the image histogram. | |

### Edge detection

<img src="https://d37oebn0w9ir6a.cloudfront.net/account_16771/1_81483def799e1d74239753e30b922307.gif" alt="Thresholding example" width="50%" />

| Name      | Description | Parameters  |
|-----------|-------------|-------------|
| **Sobel** | Detect borders in one direction using a joint Gausssian smoothing plus differentiation operation | Kernel size, direction |
| **Laplacian** | Detect borders by calculating the Laplacian of the image | |
| **Canny** | Edge Detection robust under noise developed by John F. Canny | |

### Smoothing

<img src=" https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Cappadocia_Gaussian_Blur.svg/450px-Cappadocia_Gaussian_Blur.svg.png?20150723054959" alt="Gaussian blur example" width="50%" />

| Name                | Description |
|---------------------|-------------|
| **Box blur**        | Takes the average of all the pixels under the kernel area and replaces the central element |
| **Gaussian blur**   | Bluring using a guassian kernel |
| **Median bluring**  | Replace the pixel with the median of all the pixels under the kernel area per channel |
| **Bilateral Filtering** | Like gaussian blur but keeping edges sharp |

### Sharpening

<img src="https://upload.wikimedia.org/wikipedia/commons/4/43/Unsharped_eye.jpg?20070517151252" alt="Example eye image sharpening" width="50%" />





| altName | Description |
|---------------------|-------------|
| **Unsharp masking** | Applies a Gaussian blur to a copy of the original image and substract it from the original |

### Morphology

<img src="https://www.researchgate.net/profile/Jianfei-Cai/publication/270280273/figure/fig5/AS:614028015583239@1523407095311/Morphological-operations-erosion-and-dilation-to-generate-the-trimap.png" alt="Erosion and Dilation example" width="50%" />

| Name                       | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| **Erosion**                | Shrinks the boundary of the foreground object                |
| **Dilation**               | Grows the boundary of the foreground object. Oposite of erotion |
| **Opening**                | Erosion followed by dilation (usually to remove noise in the background object) |
| **Closing**                | Dilation followed by erosion  (usually to fill small holes in the foreground object) |
| **Morphological Gradient** | Difference between dilation and erosion (usually to outline the foreground) |
| **Top hat**                | Difference between the opening of the input image and input image |
| **Black hat**              | Difference between the closing of the input image and input image |

## Material

### :books: Books

* [Digital Image Processing](https://books.google.com.ar/books/about/Digital_Image_Processing.html?id=a62xQ2r_f8wC&redir_esc=y)
* [Learning OpenCV](https://www.oreilly.com/library/view/learning-opencv/9780596516130/)
