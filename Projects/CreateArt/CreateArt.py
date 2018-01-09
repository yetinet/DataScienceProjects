from helperFunctions import *
from nstFunctions import *

def main():
    content_image = sys.argv[1:][0]
    style_image = sys.argv[1:][1]


    model_nn(content_image, style_image)

    return


if __name__== "__main__":
  main()
