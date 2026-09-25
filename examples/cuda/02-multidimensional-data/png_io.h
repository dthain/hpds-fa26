#pragma once

/*
 * Small host-only PNG I/O helper used by the image lessons in this directory.
 *
 * PNG files are compressed and may use several channel layouts. CUDA kernels
 * should not need to understand that file format. These functions ask
 * libpng's simplified API to decode a PNG into an ordinary row-major byte
 * array before CUDA sees it, or encode such an array after CUDA finishes.
 *
 * read_rgb_png:       returns R, G, B, R, G, B, ... (three bytes per pixel).
 * read_grayscale_png: returns one intensity byte per pixel.
 * write_grayscale_png writes one intensity byte per pixel as a grayscale PNG.
 */

#include <png.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

inline unsigned char* read_png_as(const char* path, int* width, int* height, png_uint_32 format) {
    png_image image = {};
    image.version = PNG_IMAGE_VERSION;

    if (!png_image_begin_read_from_file(&image, path)) {
        fprintf(stderr, "Cannot open input PNG: %s\n", image.message);
        png_image_free(&image);
        exit(EXIT_FAILURE);
    }
    image.format = format;
    const int channels = PNG_IMAGE_PIXEL_CHANNELS(format);
    if (image.width == 0 || image.height == 0 || image.width > INT_MAX / channels ||
        image.height > INT_MAX / channels / image.width) {
        fprintf(stderr, "Image dimensions exceed the integer range used by the CUDA kernels\n");
        png_image_free(&image);
        exit(EXIT_FAILURE);
    }

    unsigned char* pixels = (unsigned char*)malloc(PNG_IMAGE_SIZE(image));
    if (!pixels) {
        fprintf(stderr, "Cannot allocate image array\n");
        png_image_free(&image);
        exit(EXIT_FAILURE);
    }
    if (!png_image_finish_read(&image, NULL, pixels, 0, NULL)) {
        fprintf(stderr, "Cannot decode input PNG: %s\n", image.message);
        free(pixels);
        png_image_free(&image);
        exit(EXIT_FAILURE);
    }

    *width = (int)image.width;
    *height = (int)image.height;
    png_image_free(&image);
    return pixels;
}

inline unsigned char* read_rgb_png(const char* path, int* width, int* height) {
    return read_png_as(path, width, height, PNG_FORMAT_RGB);
}

inline unsigned char* read_grayscale_png(const char* path, int* width, int* height) {
    return read_png_as(path, width, height, PNG_FORMAT_GRAY);
}

inline void write_grayscale_png(const char* path, const unsigned char* pixels, int width, int height) {
    if (width <= 0 || height <= 0 || width > INT_MAX / height) {
        fprintf(stderr, "Invalid grayscale PNG dimensions\n");
        exit(EXIT_FAILURE);
    }

    png_image image = {};
    image.version = PNG_IMAGE_VERSION;
    image.width = width;
    image.height = height;
    image.format = PNG_FORMAT_GRAY;

    if (!png_image_write_to_file(&image, path, 0, pixels, 0, NULL)) {
        fprintf(stderr, "Cannot write output PNG: %s\n", image.message);
        png_image_free(&image);
        exit(EXIT_FAILURE);
    }
    png_image_free(&image);
}
