from PIL import Image
from IPython.display import display, Image as IPImage
from airpollutionlevels.config import resolve_path

def display_gif(file_path = resolve_path('airpollutionlevels/raw_data/animation.gif')):
    """
    Loads and displays a GIF file.

    Parameters:
        file_path (str): The path to the GIF file to be displayed.
    """
    try:
        # Open the GIF file using PIL
        gif = Image.open(file_path)

        # Check if the file is indeed a GIF
        if gif.format == 'GIF':
            print(f"Displaying GIF: {file_path}")

            display(IPImage(filename=file_path))
        else:
            print("The file is not a GIF format.")

    except Exception as e:
        print(f"An error occurred: {e}")
