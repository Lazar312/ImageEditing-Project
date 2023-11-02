#include <iostream>
#include <fstream>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <emmintrin.h>
#include <immintrin.h>
#include <chrono>
#include <ctime>
#include "timer.h"
#include "avs_mathfun.h"
#include <windows.h>
#include <cstring>

#define MAX_INT 8,589,934,592
using namespace std;

//cache 6MB set associativity 12
union VectorUnion {
    __m256i vect;
    unsigned char array[32];
};
union VectorUnionI {
    __m256i vect;
    int16_t array[32];
};
void addConstantOptimized(unsigned char *image, char constant , int size){
    __m256i vect;
    int i = 0;
    for(i = 0; i < size; i+= 32) {
        vect = _mm256_lddqu_si256((const __m256i *) (image + i));
        __m256i constantVector = _mm256_set1_epi8(constant);
        vect = _mm256_add_epi8(vect, constantVector);
        _mm256_store_si256(reinterpret_cast<__m256i *>(image + i), vect);
    }
    for(int j = i - 32; j < size; j++){
        image[i] += constant;
    }
}

void addConstant(unsigned char *image, int constant , int size){
    for(int i = 0; i < size ; i++){
        image[i] += constant;
    }
}

void subFromConstantOptimized(unsigned char* image, char constant, int size){
    __m256i vect;
    int i = 0;
    for(i = 0; i < size; i+= 32) {
        vect = _mm256_lddqu_si256((const __m256i *) (image + i));
        __m256i constantVector = _mm256_set1_epi8(constant);
        constantVector = _mm256_sub_epi8(constantVector, vect);
        _mm256_store_si256(reinterpret_cast<__m256i *>(image + i), constantVector);
    }
    for(int j = i - 32; j < size; j++){
        image[i] = constant - image[i];
    }
}

void subFromConstant(unsigned char *image, int constant , int size){
    for(int i = 0; i < size ; i++){
        image[i] = constant - image[i];
    }
}

void subConstantOptimized(unsigned char* image, char constant, int size){
    __m256i vect;
    int i = 0;
    for(i = 0; i < size; i+= 32) {
        vect = _mm256_lddqu_si256((const __m256i *) (image + i));
        __m256i constantVector = _mm256_set1_epi8(constant);
        vect = _mm256_sub_epi8(vect, constantVector);
        _mm256_store_si256(reinterpret_cast<__m256i *>(image + i), vect);
    }
    for(int j = i - 32; j < size; j++){
        image[i] -= constant;
    }
}

void subConstant(unsigned char *image, int constant , int size){
    for(int i = 0; i < size ; i++){
        image[i] -= constant;
    }
}
void multiplyWithConstantOptimized(unsigned char *image, int constant , int size){
    __m256i constantVector = _mm256_set1_epi16(constant);
    int i = 0;
    for(i = 0; i < size; i+= 16) {
        __m128i vectorAData = _mm_loadu_si128((__m128i*)(image + i ));
        __m256i vectorAData16 = _mm256_cvtepu8_epi16(vectorAData);
        __m256i result16 = _mm256_mullo_epi16(vectorAData16, constantVector);
        __m128i result = _mm_set_epi8( (_mm256_extract_epi16(result16, 15)), _mm256_extract_epi16(result16, 14), _mm256_extract_epi16(result16, 13), _mm256_extract_epi16(result16, 12),
                                       _mm256_extract_epi16(result16, 11), _mm256_extract_epi16(result16, 10), _mm256_extract_epi16(result16, 9), _mm256_extract_epi16(result16, 8),
                                       _mm256_extract_epi16(result16, 7), _mm256_extract_epi16(result16, 6), _mm256_extract_epi16(result16, 5), _mm256_extract_epi16(result16, 4),
                                       _mm256_extract_epi16(result16, 3), _mm256_extract_epi16(result16, 2), _mm256_extract_epi16(result16, 1), _mm256_extract_epi16(result16, 0)
        );
        _mm_store_si128((__m128i*)(image + i), result);
    }
    for(int j = i - 16; j < size; j++){
        image[i] *= constant;
    }
}

void multiplyWithConstant(unsigned char *image, int constant , int size){
    for(int i = 0; i < size ; i++){
        image[i] *= constant;
    }
}

void divWithConstantOptimized(unsigned char *image, int constant , int size){
    __m256 constantVector = _mm256_set1_ps((float)constant);
    __m256i vectorAData;
    int i = 0;
    for(i = 0; i < size ; i+= 8) {
        vectorAData = _mm256_cvtepu16_epi32(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(image + i))));
        __m256 vectorADataF = _mm256_cvtepi32_ps(vectorAData);
        __m256 result32 = _mm256_div_ps(constantVector, vectorADataF);
        __m256i restult32Covnverted = _mm256_cvtps_epi32(result32);
        image[i + 0] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 0));
        image[i + 1] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 1));
        image[i + 2] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 2));
        image[i + 3] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 3));
        image[i + 4] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 4));
        image[i + 5] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 5));
        image[i + 6] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 6));
        image[i + 7] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 7));
    }
    for(int j = i - 16; j < size; j++){
        image[i] *= constant;
    }
}

void divFromConstant(unsigned char *image, int constant , int size){
    for(int i = 0; i < size ; i++){
        if( image[i] != 0) {
            image[i] = constant / image[i];
        }else {
            image[i] = 0;
        }
    }
}
void divConstantOptimized(unsigned char *image, int constant , int size){
    __m256 constantVector = _mm256_set1_ps(1.0/constant);
    int i = 0;
    for(i = 0; i < size; i+= 8) {
        __m128i vect = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(image + i)));
        __m256i vectorAData = _mm256_cvtepu16_epi32(vect);
        __m256 vectorADataF = _mm256_cvtepi32_ps(vectorAData);
        __m256 result32 = _mm256_mul_ps(vectorADataF, constantVector);
        __m256i restult32Covnverted = _mm256_cvtps_epi32(result32);;
        image[i + 0] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 0));
        image[i + 1] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 1));
        image[i + 2] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 2));
        image[i + 3] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 3));
        image[i + 4] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 4));
        image[i + 5] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 5));
        image[i + 6] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 6));
        image[i + 7] = ((unsigned char)_mm256_extract_epi32(restult32Covnverted, 7));
    }
    for(int j = i - 16; j < size; j++){
        image[i] *= constant;
    }
}
void divConstant(unsigned char *image, int constant , int size){
    for(int i = 0; i < size ; i++){
        image[i] /= constant;
    }
}

void pow(unsigned char *image, int constant , int size){
    for(int i = 0; i < size ; i++){
        int p = image[i];
        for(int j = 1; j < constant; j ++){
            p *= image[i];
        }
        image[i] = p;
    }
}

void powOptimized(unsigned char *image, int constant , int size){
    int i = 0;
    for(i = 0; i < size; i+= 8) {
        __m256i vectorAData = _mm256_set_epi32(image[7 + i], image[6 + i], image[ 5+ i], image[4 + i],image[i + 3], image[i + 2], image[i + 1], image [i]);
        __m256i vectorAData16Temp = vectorAData;

        int j = 0;
        for(j = 2; j < constant; j = j << 1){
            vectorAData = _mm256_mullo_epi32(vectorAData, vectorAData);
        }
        for(int n = j/2; n < constant; n++){
            vectorAData = _mm256_mullo_epi32(vectorAData, vectorAData16Temp);
        }
        image[i + 0] = ((unsigned char)_mm256_extract_epi32(vectorAData, 0));
        image[i + 1] = ((unsigned char)_mm256_extract_epi32(vectorAData, 1));
        image[i + 2] = ((unsigned char)_mm256_extract_epi32(vectorAData, 2));
        image[i + 3] = ((unsigned char)_mm256_extract_epi32(vectorAData, 3));
        image[i + 4] = ((unsigned char)_mm256_extract_epi32(vectorAData, 4));
        image[i + 5] = ((unsigned char)_mm256_extract_epi32(vectorAData, 5));
        image[i + 6] = ((unsigned char)_mm256_extract_epi32(vectorAData, 6));
        image[i + 7] = ((unsigned char)_mm256_extract_epi32(vectorAData, 7));
    }
    for(int j = i - 16; j < size; j++){
        image[i] *= constant;
    }
}
void abs(unsigned char *image, int size){
    for(int i = 0; i < size ; i++){
        if(image[i] > 128)
            image[i] = 256 - image[i];
    }
}
void min(unsigned char *image, char constant, int size){
    for(int i = 0; i < size ; i++){
        image[i] = min((char)image[i], constant);
    }
}
void minOptimized(unsigned char *image, char constant, int size){
    __m256i constVector = _mm256_set1_epi8(constant);
    for(int i = 0; i < size ; i+=32) {
        __m256i vector = _mm256_load_si256((__m256i *) (image + i));
        vector = _mm256_min_epi8(vector,constVector);
        _mm256_store_si256((__m256i *) (image + i), vector);
    }
}
void max(unsigned char *image, char constant, int size){
    for(int i = 0; i < size ; i++){
        image[i] = max((char)image[i], constant);
    }
}
void maxOptimized(unsigned char *image, char constant, int size){
    __m256i constVector = _mm256_set1_epi8(constant);
    for(int i = 0; i < size ; i+=32) {
        __m256i vector = _mm256_load_si256((__m256i *) (image + i));
        vector = _mm256_max_epi8(vector,constVector);
        _mm256_store_si256((__m256i *) (image + i), vector);
    }
}

void absOptimized(unsigned char *image , int size){
    for(int i = 0; i < size ; i+=32) {
        __m256i vector = _mm256_load_si256((__m256i *) (image + i));
        vector = _mm256_abs_epi8(vector);
        _mm256_store_si256((__m256i *) (image + i), vector);
    }
}
void inverse(unsigned char *image, int size){
    for(int i = 0; i < size ; i++){
        image[i] = 256 - image[i];
    }
}
void inverseOptimized(unsigned char *image, int size){
    __m256i constVector = _mm256_set1_epi8(256);
    for(int i = 0; i < size ; i+=32) {
        __m256i vector = _mm256_load_si256((__m256i *) (image + i));
        vector = _mm256_sub_epi8(constVector, vector);
        _mm256_store_si256((__m256i *) (image + i), vector);
    }
}

void blackAndWhiteIMG(unsigned char *image, int size){
    for (int i = 0; i < size; i +=3){
        float R = image[i] * 0.299f;
        float G = image[i + 1] * 0.587f;
        float B = image[i + 2] * 0.114f;
        unsigned char I = (R + G + B);
        image[i] = I;
        image[i + 1] = I;
        image[i + 2] = I;
    }
}
void blackAndWhiteIMGOptimized(unsigned char *image, int size){
    __m256 cR = _mm256_set1_ps(0.299f);
    __m256 cG = _mm256_set1_ps(0.587f);
    __m256 cB = _mm256_set1_ps(0.114f);
    __m256 R, G, B, mR, sum;
    for (int i = 0; i < size;){
        R = _mm256_cvtepi32_ps(_mm256_setr_epi32(image[i], image[i + 3], image[i + 6], image[i + 9], image[i + 12], image[i + 15], image[i + 18], image[i + 21]));
        G = _mm256_cvtepi32_ps(_mm256_setr_epi32(image[i + 1], image[i + 4], image[i + 7], image[i + 10], image[i + 13], image[i + 16], image[i + 19], image[i + 22]));
        B = _mm256_cvtepi32_ps(_mm256_setr_epi32(image[i + 2], image[i + 5], image[i + 8], image[i + 11], image[i + 14], image[i + 17], image[i + 20], image[i + 23]));
        mR = _mm256_mul_ps(R, cR);
        sum = _mm256_fmadd_ps(G, cG, mR);
        sum = _mm256_fmadd_ps(B, cB, sum);
        __m256i sumInt = _mm256_cvtps_epi32(sum);
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 0));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 0));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 0));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 1));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 1));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 1));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 2));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 2));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 2));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 3));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 3));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 3));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 4));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 4));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 4));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 5));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 5));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 5));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 6));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 6));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 6));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 7));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 7));
        image[i++] = ((unsigned char)_mm256_extract_epi32(sumInt, 7));
    }
}

void log(unsigned char *image, int size){
    for(int i = 0; i < size; i++){
        image[i] = (unsigned char) (log(image[i]) / log(2.71));
    }
}
void logOptimized(unsigned char *image, int size){
    __m128i vector;
    int i = 0;
    for(i = 0; i < size ; i+=8) {

        vector = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(image + i)));
        __m256i vectorAData =_mm256_cvtepu16_epi32(vector);
        __m256 vectorADataF = _mm256_cvtepi32_ps(vectorAData);
        vectorADataF = *log256_ps(&vectorADataF);
        __m256i convertResult = _mm256_cvtps_epi32(vectorADataF);
        image[i + 0] = ((unsigned char)_mm256_extract_epi32(convertResult, 0));
        image[i + 1] = ((unsigned char)_mm256_extract_epi32(convertResult, 1));
        image[i + 2] = ((unsigned char)_mm256_extract_epi32(convertResult, 2));
        image[i + 3] = ((unsigned char)_mm256_extract_epi32(convertResult, 3));
        image[i + 4] = ((unsigned char)_mm256_extract_epi32(convertResult, 4));
        image[i + 5] = ((unsigned char)_mm256_extract_epi32(convertResult, 5));
        image[i + 6] = ((unsigned char)_mm256_extract_epi32(convertResult, 6));
        image[i + 7] = ((unsigned char)_mm256_extract_epi32(convertResult, 7));
    }
    for(int j = i; j < size; j++){
        image[j] = (unsigned char) (log(image[j]) / log(2.71));
    }
}
void filtrate(unsigned char* image, int width, int height, float **filter, int filterSize) {

    unsigned char* tempImage = new unsigned char[width * height * 3];
    std::memcpy(tempImage, image, width * height * 3);
    for (int x = filterSize; x < width - filterSize; x++) {
        for (int y = filterSize; y < height - filterSize; y++) {
            for (int c = 0; c < 3; c++) {
                int sum = 0;
                for (int i = -filterSize; i <= filterSize; i++) {
                    for (int j = -filterSize; j <= filterSize; j++) {
                        int pixelIndex = ((y + i) * width + (x + j)) * 3;
                        sum += tempImage[pixelIndex + c] * filter[i + +filterSize][j + filterSize];
                    }
                }

                int pixelIndex = (y * width + x) * 3;
                image[pixelIndex + c] = std::min(std::max(sum, 0), 255);
            }
        }
    }

    delete[] tempImage;
}

void filtrateOptimized(unsigned char* image, int width, int height, float **filter, int filterSize, int cache_line, int blockNum) {
    unsigned char* tempImage = new unsigned char[width * height * 3];
    std::memcpy(tempImage, image, width * height * 3);
    for (int yy = filterSize; yy < height - filterSize; yy+=cache_line) {
        for (int xx = filterSize; xx < width - filterSize; xx += cache_line ) {
            for (int y = yy; y < height - filterSize && y < yy + cache_line; y++) {
                for (int x = xx; x < width - filterSize && x < xx + cache_line; x++) {
                        int sum = 0;
                        for (int i = -filterSize; i <= filterSize; i++) {
                            for (int j = -filterSize; j <= filterSize; j++) {
                                int pixelIndex = ((y + i) * width + (x + j)) * 3;
                                sum += tempImage[pixelIndex ] * filter[i + +filterSize][j + filterSize];
                            }
                        }
                        int pixelIndex = (y * width + x) * 3;
                        image[pixelIndex ] = std::min(std::max(sum, 0), 255);
                        sum = 0;
                        for (int i = -filterSize; i <= filterSize; i++) {
                            for (int j = -filterSize; j <= filterSize; j++) {
                                int pixelIndex = ((y + i) * width + (x + j)) * 3;
                                sum += tempImage[pixelIndex + 1] * filter[i + +filterSize][j + filterSize];
                            }
                        }
                        pixelIndex = (y * width + x) * 3;
                        image[pixelIndex + 1] = std::min(std::max(sum, 0), 255);
                        sum = 0;
                        for (int i = -filterSize; i <= filterSize; i++) {
                            for (int j = -filterSize; j <= filterSize; j++) {
                                int pixelIndex = ((y + i) * width + (x + j)) * 3;
                                sum += tempImage[pixelIndex + 2] * filter[i + +filterSize][j + filterSize];
                            }
                        }
                        pixelIndex = (y * width + x) * 3;
                        image[pixelIndex + 2] = std::min(std::max(sum, 0), 255);
                }
            }
        }
    }
    delete[] tempImage;
}


int main() {
    DWORD bufferSize = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;

    GetLogicalProcessorInformation(NULL, &bufferSize);

    buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(bufferSize);
    if (buffer == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    if (!GetLogicalProcessorInformation(buffer, &bufferSize)) {
        printf("Failed to retrieve cache information.\n");
        free(buffer);
        return 1;
    }

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
    DWORD offset = 0;
    unsigned short cache_line;
    unsigned int size;
    unsigned int blkNo;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= bufferSize) {
        if (ptr->Relationship == RelationCache && ptr->Cache.Level == 1 && ptr->Cache.Type == 2) {
            cache_line = ptr->Cache.LineSize;
            size =  ptr->Cache.Size;
            blkNo = size/cache_line;
            break;
        }

        offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;


    }
    free(buffer);

    while(true){
        int q;
        int x, y, n;
        int x1,y1,n1;

        //unsigned char* img __attribute__((aligned(32)))  = stbi_load("../slika1.jpg",&x,&y,&n,0);
        unsigned char* img __attribute__((aligned(32)))  = stbi_load("../slika2.bmp",&x,&y,&n,0);
        int size = n * y * x;
        if(!img){
            cout << "No image found";
            exit(-1);
        }
        //unsigned char* img2 __attribute__((aligned(32)))  = stbi_load("../slika1.jpg",&x1,&y1,&n1,0);
        unsigned char* img2 __attribute__((aligned(32)))  = stbi_load("../slika2.bmp",&x1,&y1,&n1,0);
        int size2 = n1 * y1 * x1;
        if(!img2){
            cout << "No image found";
            exit(-1);
        }
        int constant;
        cout << "Enter a number for specific optimisation" << endl;
        cout << "1 for add constant" << endl;
        cout << "2 for sub constant" << endl;
        cout << "3 for subbing from constant" << endl;
        cout << "4 for multipling" << endl;
        cout << "5 for division" << endl;
        cout << "6 for inverse devision" << endl;
        cout << "7 for power instruction" << endl;
        cout << "8 for log" << endl;
        cout << "9 for abs" << endl;
        cout << "10 for min" << endl;
        cout << "11 for max" << endl;
        cout << "12 for inversion" << endl;
        cout << "13 for black and white image" << endl;
        cout << "14 for aplying a filter" << endl;
        cout << "15 for exit" << endl;
        cin >> q;

        switch (q) {
            case 1:
                cout << "Enter a constant" << endl;
                cin >> constant;
                StartTimer(No SIMD);
                    addConstant(img,constant,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    addConstantOptimized(img2,constant,size2);
                EndTimer;
                break;
            case 2:
                cout << "Enter a constant" << endl;
                cin >> constant;
                StartTimer(No SIMD);
                    subConstant(img,constant,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    subConstantOptimized(img2,constant,size2);
                EndTimer;
                break;
            case 3:
                cout << "Enter a constant" << endl;
                cin >> constant;
                StartTimer(No SIMD);
                    subFromConstant(img,constant,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    subFromConstantOptimized(img2,constant,size2);
                EndTimer;
                break;
            case 4:
                cout << "Enter a constant" << endl;
                cin >> constant;
                StartTimer(No SIMD);
                    multiplyWithConstant(img,constant,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    multiplyWithConstantOptimized(img2,constant,size2);
                EndTimer;
                break;
            case 5:
                cout << "Enter a constant" << endl;
                cin >> constant;
                StartTimer(No SIMD);
                    divConstant(img,constant,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    divConstantOptimized(img2,constant,size2);
                EndTimer;
                break;
            case 6:
                cout << "Enter a constant" << endl;
                cin >> constant;
                StartTimer(No SIMD);
                    divFromConstant(img,constant,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    divWithConstantOptimized(img2,constant,size2);
                EndTimer;
                break;
            case 7:
                cout << "Enter a constant" << endl;
                cin >> constant;
                StartTimer(No SIMD);
                    pow(img,constant,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    powOptimized(img2,constant,size2);
                EndTimer;
                break;
            case 8:
                StartTimer(No SIMD);
                    log(img,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    logOptimized(img2,size2);
                EndTimer;
                break;
            case 9:
                StartTimer(No SIMD);
                    abs(img,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    absOptimized(img2,size2);
                EndTimer;
                break;
            case 10:
                cout << "Enter a constant" << endl;
                cin >> constant;
                StartTimer(No SIMD);
                    min(img,constant,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    minOptimized(img2,constant,size2);
                EndTimer;
                break;
            case 11:
                cout << "Enter a constant" << endl;
                cin >> constant;
                StartTimer(No SIMD);
                    max(img,constant,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    max(img2,constant,size2);
                EndTimer;
                break;
            case 12:
                StartTimer(No SIMD);
                    inverse(img,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    inverseOptimized(img2,size2);
                EndTimer;
                break;
            case 13:
                StartTimer(No SIMD);
                    blackAndWhiteIMG(img,size);
                EndTimer;
                StartTimer(Wt SIMD);
                    blackAndWhiteIMGOptimized(img2,size2);
                EndTimer;
                break;
            case 14:
                int N;
                cout << "Enter filter size" << endl;
                cin >> N;
                while(N %2 == 0){
                    cout << "Enter valid size N%2 == 1" << endl;
                    cin >> N;
                }
                float **filter;
                filter = new  float* [N];
                for(int i = 0; i < N; i++){
                    filter[i] = new float [N];
                }
                for(int i = 0; i < N; i++){
                    for(int j = 0; j < N ; j ++){
                        cout << "(" << i << ", " << j << "):" ;
                        cin >> filter[i][j];
                    }
                }
                StartTimer(No SIMD);
                    filtrate(img,x, y, filter, N/2);
                EndTimer;
                StartTimer(Wt SIMD);
                    filtrateOptimized(img2,x1, y1, filter, N/2, cache_line, blkNo);
                EndTimer;
                for (int i = 0; i < N; ++i)
                    delete [] filter[i];
                delete [] filter;
                break;
            default:
                break;
        }

        //stbi_write_jpg("../sky1.jpg",x,y,n,img,0);
        stbi_write_bmp("../sky1.bmp",x,y,n,img);
        stbi_image_free(img);
        //stbi_write_jpg("../sky2.jpg",x1,y1,n1,img2,0);
        stbi_write_bmp("../sky2.bmp",x1,y1,n1,img2);
        stbi_image_free(img2);
        if( q == 15)break;
        q = 0;
    }


}