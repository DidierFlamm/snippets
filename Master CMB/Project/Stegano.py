# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:18:22 2023

@author: Didier
"""

from PIL import Image

def string_to_binary(message):
    """converts a string message into binary with constant length 8 bits"""
    return(''.join(format(ord(char), '08b') for char in message))
    #ord converts every character to binary using Unicode
    #format with parameter '08b' make Unicode values have a constant length = 8
           
def binary_to_string(binary_message):
    message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i + 8]
        message += chr(int(byte, 2))
    return(message)
           
# Encode a message into an image
def encode_image(input_image_path, output_image_path, message):
    # Open the original image
    image = Image.open(input_image_path)
    width, height = image.size
    channels = len(image.getbands()) #3 for RGB or 1 for gray-scale image

    # Convert the message to binary
    binary_message = string_to_binary(message)
    
    # Add 'þ' as stop code, = 254 according to Unicode 
    binary_message += string_to_binary('þ')
        
    # Make sure the message can fit within the image
    if len(binary_message) > width * height * channels:
        raise ValueError("Message is too large to encode in the image.")

    data_index = 0
    for y in range(height):
        for x in range(width):
            
            pixel = image.getpixel((x, y)) #extract value(s) of pixel(x,y)
            if channels == 1:
                pixel = [pixel,] #converts integer value of gray-scale to list
            else:
                pixel = list(image.getpixel((x, y))) # type: ignore # converts tuple into list to make it mutable

            for color_channel in range(channels):  # 3 if RGB channels or 1 if gray-scale image
                if data_index < len(binary_message):
                    # Modify the LSB of the color channel to encode the message
                    pixel[color_channel] = pixel[color_channel] & 254 | int(binary_message[data_index]) # type: ignore
                    data_index += 1
            image.putpixel((x, y), tuple(pixel)) # type: ignore #putpixel modify a pixel by the values of a tuple
            
            if data_index == len(binary_message):
                break
        
        if data_index == len(binary_message):
            break

    # Save the encoded image
    image.save(output_image_path)

# Decode the hidden message from an image
def decode_image(input_image_path):
    image = Image.open(input_image_path)
    width, height = image.size
    channels = len(image.getbands()) #3 for RGB or 1 for gray-scale image
    
    binary_message = ""

    for y in range(height):
        for x in range(width):
            
            pixel = image.getpixel((x, y))
            if channels == 1:
                pixel = (pixel,) #convert value to tuple
            
            for color_channel in range(channels):  # RGB channels
                binary_message += str(pixel[color_channel] & 1) # type: ignore #get the last bit of every pixel value
            

    message = ""
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i + 8]
        message += chr(int(byte, 2))
        
        if int(byte, 2) == 254: #254 is the stop code
            break

    return message[:-1] #doesn't return stop code

# Example usage
input_image_path = "lena512.png" # "lena512.png" or  "baboon.jpg"
output_image_path = "output_image.png" #do NOT use extension .jpg car compress image and modify pixels
message_to_hide = "This is the message to hide"

encode_image(input_image_path, output_image_path, message_to_hide)

decoded_message = decode_image(output_image_path)
print("Decoded Message:", decoded_message)

