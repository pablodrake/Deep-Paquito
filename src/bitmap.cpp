/* 
 * File: src/bitmap.cpp
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains functionality for working with bitmap images.
 */
#include <filesystem>
#include <fstream>
#include "bitmap.h"


using namespace std;


unsigned char *** reserveSpaceImage(int height, int width) {

    unsigned char *** image = new unsigned char **[height];
    for(int i = 0; i < height; i++) {
        image[i] = new unsigned char * [width];
        for(int j = 0; j < width; j++) {
            image[i][j] = new unsigned char[BYTES_PER_PIXEL];
        }
    }

    return image;
}



void freeSpaceImage(unsigned char ***image, int height, int width) {

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            delete[] image[i][j];
        }
        delete[] image[i];
    }
    delete[] image;
}



void generateBitmapImage(unsigned char*** image, int height, int width, char* image_file_name) {
    // Width of the image in bytes
    int width_in_bytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3] = {0, 0, 0};  // It's 3 bytes because if it were 4
                                           // it would already be aligned, and padding wouldn't be needed
    int padding_size = (4 - width_in_bytes % 4) % 4;

    // stride is the space between two rows
    int stride = width_in_bytes + padding_size;

    FILE* image_file = fopen(image_file_name, "wb");

    // Create file header and write it to the file
    unsigned char* file_header = createBitmapFileHeader(height, stride);
    fwrite(file_header, 1, FILE_HEADER_SIZE, image_file);

    // Create info header and write it to the file
    unsigned char* info_header = createBitmapInfoHeader(height, width);
    fwrite(info_header, 1, INFO_HEADER_SIZE, image_file);

    // Write the channels of each pixel next to each other and the padding
    // The shape of the data will be [height, width, channels]

    // Writing in reverse order because the first row is the bottom of the image
    for(int i = height - 1; i >= 0; i--) {
        for(int j = 0; j < width; j++) {
            // pixels
            fwrite(image[i][j], BYTES_PER_PIXEL, 1, image_file);
        }
        // padding at the end of each line to make it a multiple of 4
        fwrite(padding, 1, padding_size, image_file);
    }

    fclose(image_file);
}

// 14 bytes file header
unsigned char* createBitmapFileHeader(int height, int stride) {
    int file_size = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char file_header[] = {
        0,0,    
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
    };

    // Bitmap file signature
    file_header[0] = (unsigned char)('B');
    file_header[1] = (unsigned char)('M');
    // We have to put the file size in 4 bytes (4 unsigned char),
    // because only the first byte of the int is taken when typecasting
    file_header[2] = (unsigned char)(file_size);
    file_header[3] = (unsigned char)(file_size >> 8);
    file_header[4] = (unsigned char)(file_size >> 16);
    file_header[5] = (unsigned char)(file_size >> 24);

    // Although we have 4 bytes to specify the size of the headers, we only need one
    file_header[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return file_header;
}

// 40 bytes info header
unsigned char* createBitmapInfoHeader(int height, int width) {
    static unsigned char info_header[] = {
        0,0,0,0,          // header size
        0,0,0,0,          // image width
        0,0,0,0,          // image height
        0,0,              // number of color planes
        0,0,              // bits per pixel
        0,0,0,0,          // compression
        0,0,0,0,          // image size
        0,0,0,0,          // horizontal resolution
        0,0,0,0,          // vertical resolution
        0,0,0,0,          // colors in color table
        0,0,0,0,          // color count
    };

    // Info header size < 255
    info_header[0]  = (unsigned char)(INFO_HEADER_SIZE);
    // Image width
    info_header[4]  = (unsigned char)(width);
    info_header[5]  = (unsigned char)(width >> 8);
    info_header[6]  = (unsigned char)(width >> 16);
    info_header[7]  = (unsigned char)(width >> 24);
    // Image height
    info_header[8]  = (unsigned char)(height);
    info_header[9]  = (unsigned char)(height >> 8);
    info_header[10] = (unsigned char)(height >> 16);
    info_header[11] = (unsigned char)(height >> 24);
    // According to the specification, it always has to be 1
    info_header[12] = (unsigned char)(1);
    // Number of bits per pixel
    info_header[14] = (unsigned char)(BYTES_PER_PIXEL * 8);

    return info_header;
}