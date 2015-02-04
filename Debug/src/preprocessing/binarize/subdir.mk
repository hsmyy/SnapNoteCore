################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/preprocessing/binarize/binarize.cpp 

OBJS += \
./src/preprocessing/binarize/binarize.o 

CPP_DEPS += \
./src/preprocessing/binarize/binarize.d 


# Each subdirectory must supply rules for building sources it contributes
src/preprocessing/binarize/%.o: ../src/preprocessing/binarize/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include -I/usr/local/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


