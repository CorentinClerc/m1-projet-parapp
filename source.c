#include <stdio.h>
// #include <stdlib.h>
#include <omp.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "lib_stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib_stb_image/stb_image_write.h"

unsigned char *saveAsGreyscaleParallel(int imgWidth, int imgHeight, int channels, unsigned char *img, char *savePath){
    // Adds alpha channel if relevant
    int greyChannels = channels == 4 ? 2 : 1;
    // Assigns the image pointer
    unsigned char *greyscaleImg = (unsigned char *)malloc(imgWidth * imgHeight * greyChannels * sizeof(unsigned char));
    
    #pragma omp parallel for schedule(dynamic, 16)
    for(int i = 0; i < imgHeight * imgWidth; i += 1){
        // In the context of a parallel loop, we need to define the pixel as an offset of i channels in the image 
        // To avoid any dependency between loop executions
        unsigned char *pixel = img + i * channels;          // no need for the x and y coordinates here, so we will simply check pixels like in a 1d array
        unsigned char *greyPixel = greyscaleImg + i * greyChannels;
        // The actual pixel color calculation
        *greyPixel = (uint8_t)((*pixel * 0.299) + (*(pixel + 1) * 0.587) + (*(pixel + 2) * 0.114));
        
        // If we have an alpha channel, we simply set the value at the one found in the original image
        if(greyChannels > 1){
            *(greyPixel + 1) = (uint8_t)(*(pixel + 3));
        }
    }
    // We save the image and return it's binary
    // stbi_write_png(savePath, imgWidth, imgHeight, greyChannels, greyscaleImg, imgWidth * greyChannels);
    // free(greyscaleImg);
    return greyscaleImg;
}

long long calculateDifferenceScore(
    unsigned char *inImg, int inWidth, int inHeight,
    unsigned char *schImg, int schWidth, int schHeight,
    int xStart, int yStart
){
    // We assume both images only have one channel
    long long score = 0;
    // We can collapse as we can calculate each pixel independently from one another
    #pragma omp parallel for collapse(2) // reduce(+:score)
    for(int x = 0; x < schHeight; x++){
        for(int y = 0; y < schWidth; y++){
            unsigned char *schPixel = schImg + (x * schWidth + y) * 2;  // alpha channel in the search img
            unsigned char *inPixel = inImg + ((x + xStart) * inWidth + (y + yStart));
            long long value = ((*inPixel) - (*schPixel)) * ((*inPixel) - (*schPixel));
            // The calculation can be done at anytime by a thread, but the score increments needs to be synched in order to avoid multiple threads from try to access it at the same time.
            // #pragma omp critical
            score += value;
        }
    }
    return score;
}

int CompleteSSD(){

    // Get image paths from arguments.
    const char *inputImgPath = "img/space.png";
    const char *searchImgPath = "img/goat.png";

    // ==================================== Loading input image.
    #pragma region 
    int inputImgWidth;
    int inputImgHeight;
    int dummyNbChannels; // number of channels forced to 3 in stb_load.
    unsigned char *inputImg = stbi_load(inputImgPath, &inputImgWidth, &inputImgHeight, &dummyNbChannels, 3);
    if (inputImg == NULL)
    {
        printf("Cannot load image %s", inputImgPath);
        return EXIT_FAILURE;
    }
    printf("Input image %s: %dx%d\n", inputImgPath, inputImgWidth, inputImgHeight);
    #pragma endregion
    // ====================================  Loading search image.
    #pragma region 
    int searchImgWidth;
    int searchImgHeight;
    unsigned char *searchImg = stbi_load(searchImgPath, &searchImgWidth, &searchImgHeight, &dummyNbChannels, 4);
    if (searchImg == NULL)
    {
        printf("Cannot load image %s", searchImgPath);
        return EXIT_FAILURE;
    }
    printf("Search image %s: %dx%d\n", searchImgPath, searchImgWidth, searchImgHeight);
    printf("Loading complete\n");
    #pragma endregion
    // ==================================== Creating the two Greyscales
    #pragma region
    printf("Turning images to greyscale... \n");
    unsigned char *greyInput = saveAsGreyscaleParallel(inputImgWidth, inputImgHeight, 3, inputImg, "img/grey_input.png");
    unsigned char *greySearch = saveAsGreyscaleParallel(searchImgWidth, searchImgHeight, 4, searchImg, "img/grey_search.png");
    printf("Greyscale complete\n");
    #pragma endregion
    // ====================================  Looping through the input greyscaled
    #pragma region
    // We will create an array of integer that each represent a score achieved by a subsection of the input image from a pixel
    printf("Visualisation of search area...\n");
    unsigned char *searchArea = (unsigned char *)malloc(inputImgWidth * inputImgHeight * 3 * sizeof(unsigned char));

    int xBest = 0;
    int yBest = 0;
    long long bestScore = LONG_LONG_MAX;

    #pragma omp parallel for collapse(2)
    for(int x = 0; x < inputImgHeight; x += 1){
        for(int y = 0; y < inputImgWidth; y += 1){
            int i = (x * inputImgWidth + y);
            unsigned char *searchAreaPixel = searchArea + i * 3;
            unsigned char *basePixel = inputImg + i * 3;
            *searchAreaPixel = (uint8_t)(*basePixel);
            *(searchAreaPixel + 1) = (uint8_t)(*(basePixel + 1));
            *(searchAreaPixel + 2) = (uint8_t)(*(basePixel + 2)); 
            
            if(x < inputImgHeight - searchImgHeight
                && y < inputImgWidth - searchImgWidth){
                long long score = calculateDifferenceScore(
                    greyInput, inputImgWidth, inputImgHeight,
                    greySearch, searchImgWidth, searchImgHeight,
                    x, y);
                
                if(score < bestScore){
                    *(searchAreaPixel + 2) = 255;
                    bestScore = score;
                    xBest = x;
                    yBest = y;
                }
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for(int x = xBest; x < xBest + searchImgHeight; x++ ){
        unsigned char *searchAreaPixel = searchArea + (x * inputImgWidth + yBest) * 3;    
        *searchAreaPixel = 255;
        *(searchAreaPixel+1) = 0;
        *(searchAreaPixel+2) = 0;

        searchAreaPixel = searchArea + (x * inputImgWidth + yBest + searchImgWidth) * 3;
        *searchAreaPixel = 255;
        *(searchAreaPixel+1) = 0;
        *(searchAreaPixel+2) = 0;
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for(int y = yBest; y < yBest + searchImgWidth; y++){
        unsigned char *searchAreaPixel = searchArea + (xBest * inputImgWidth + y) * 3;
        *searchAreaPixel = 255;
        *(searchAreaPixel+1) = 0;
        *(searchAreaPixel+2) = 0;

        searchAreaPixel = searchArea + ((xBest + searchImgHeight) * inputImgWidth + y) * 3;
        *searchAreaPixel = 255;
        *(searchAreaPixel+1) = 0;
        *(searchAreaPixel+2) = 0;
    }

    printf("Best score : %lli at (%i,%i)", bestScore, xBest, yBest);
    #pragma endregion
    
    // ===================  Saving the results
    printf("Visualisation of search area complete\n");
    stbi_write_png("img/searchArea.png", inputImgWidth, inputImgHeight, 3, searchArea, inputImgWidth * 3);
    
    printf("Save example\n");
    unsigned char *saveExample = (unsigned char *)malloc(inputImgWidth * inputImgHeight * 1 * sizeof(unsigned char));
    memcpy( saveExample, greyInput, inputImgWidth * inputImgHeight * 1 * sizeof(unsigned char) );
    printf("Save example write\n");
    stbi_write_png("img/save_example.png", inputImgWidth, inputImgHeight, 1, saveExample, inputImgWidth * 1);
    // ===================  Freeing memory
    printf("Freeing memory\n");
    free(greyInput);
    free(greySearch);
    free(searchArea);
    free(saveExample);
    stbi_image_free(inputImg); 
    stbi_image_free(searchImg); 

    printf("Program complete. Good bye!\n");

    return EXIT_SUCCESS;
}

int BenchmarkGreyscale(){
    // We will not save anything here, simply run it multiple times
    int benchmarkCount = 100;
    int benchmarkWarmup = 5;
    const char* inputImgPath = "";
    const char* searchImgPath = "";
    #pragma region 
    int inputImgWidth;
    int inputImgHeight;
    int dummyNbChannels; // number of channels forced to 3 in stb_load.
    unsigned char *inputImg = stbi_load(inputImgPath, &inputImgWidth, &inputImgHeight, &dummyNbChannels, 3);
    if (inputImg == NULL)
    {
        printf("Cannot load image %s", inputImgPath);
        return EXIT_FAILURE;
    }
    printf("Input image %s: %dx%d\n", inputImgPath, inputImgWidth, inputImgHeight);
    #pragma endregion
    // ====================================  Loading search image.
    #pragma region 
    int searchImgWidth;
    int searchImgHeight;
    unsigned char *searchImg = stbi_load(searchImgPath, &searchImgWidth, &searchImgHeight, &dummyNbChannels, 4);
    if (searchImg == NULL)
    {
        printf("Cannot load image %s", searchImgPath);
        return EXIT_FAILURE;
    }
    printf("Search image %s: %dx%d\n", searchImgPath, searchImgWidth, searchImgHeight);
    printf("Loading complete\n");
    #pragma endregion

    for(int i = 0; i < benchmarkWarmup; i++){
        unsigned char *greyTest = saveAsGreyscaleParallel(inputImgWidth, inputImgHeight, 3, inputImg, "img/grey_input.png");
    }
    clock_t startTime = clock();
    for(int i = 0; i < benchmarkCount; i++){
        unsigned char *greyInput = saveAsGreyscaleParallel(inputImgWidth, inputImgHeight, 3, inputImg, "img/grey_input.png");
    }
    clock_t endTime = clock();
    unsigned long ellapsedMs = (endTime - startTime) * 1000 / CLOCKS_PER_SEC;
    printf("Benchmark took %li ms to complete %i iterations of greyscale.", ellapsedMs, benchmarkCount);
    return EXIT_SUCCESS;
}

int main (int argc, char *argv[])
{
    BenchmarkGreyscale();
}


