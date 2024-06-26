import os
from PIL import Image, ImageFilter
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

image_url = "../assets/images/img_2.jpg"


def img_to_ocr_using_grayscale():
    images_formats_not_supported = [".avif", ".webp", ".tiff", ".bmp", ".heif"]

    if os.path.isfile(path=image_url):
        print("The image exists.")
    else:
        print("The image does not exist.")
        return

    # If the image isn't jpg or png convert to jpg
    for not_supported in images_formats_not_supported:
        if image_url.endswith(not_supported):
            print("The image is not supported.")
            return

    # convert the image to grayscale
    grayscale_image = Image.open(image_url).convert("L")
    # Image binary
    binary_image = grayscale_image.point(lambda x: 0 if x < 128 else 255)
    # image fragment
    fragments_image = binary_image.split()
    # thin characters
    thin_characters = []
    for thin in thin_characters:
        thin_characters.append(thin.filter(ImageFilter.RankFilter(3, 1)))
    # Compare with patterns
    characters = []
    for fragment in fragments_image:
        characters.append(pytesseract.image_to_string(fragment))

    # show the text of the image
    print("".join(characters))


def img_to_ocr_without_grayscale():
    img = Image.open(image_url)
    text = pytesseract.image_to_string(img)
    print(text)


# hard example with matplotlib and numpy showing in a cartesian plane
def img_to_ocr_using_numpy():
    img = Image.open(image_url)
    img = np.array(img)
    plt.imshow(img)
    plt.show()


# show text in a cartesian plane just with points with black color avoid using the image in the top
def img_to_ocr_using():
    img = Image.open(image_url)
    img = np.array(img)
    plt.imshow(img, cmap="gray")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("With grayscale:\n")
    img_to_ocr_using_grayscale()

    print("\nWithout grayscale:\n")
    img_to_ocr_without_grayscale()

    print("\nMatplotlib:\n")
    img_to_ocr_using()
