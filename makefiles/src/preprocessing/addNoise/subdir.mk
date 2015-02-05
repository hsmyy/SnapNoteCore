################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/preprocessing/addNoise/addnoise.cpp 

OBJS += \
./src/preprocessing/addNoise/addnoise.o 

CPP_DEPS += \
./src/preprocessing/addNoise/addnoise.d 


# Each subdirectory must supply rules for building sources it contributes
src/preprocessing/addNoise/%.o: ../src/preprocessing/addNoise/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/xxy/opencv/include -I/home/xxy/opencv/include/opencv -I/home/xxy/tesseract/include -I/home/xxy/leptonica/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


