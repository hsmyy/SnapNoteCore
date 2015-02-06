################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/preprocessing/binarize/binarize.cpp \
../src/preprocessing/binarize/test.cpp 

OBJS += \
./src/preprocessing/binarize/binarize.o \
./src/preprocessing/binarize/test.o 

CPP_DEPS += \
./src/preprocessing/binarize/binarize.d \
./src/preprocessing/binarize/test.d 


# Each subdirectory must supply rules for building sources it contributes
src/preprocessing/binarize/%.o: ../src/preprocessing/binarize/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/ubuntu/opencv/include -I/home/ubuntu/opencv/include/opencv -I/home/ubuntu/tesseract/include -I/home/ubuntu/leptonica/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


