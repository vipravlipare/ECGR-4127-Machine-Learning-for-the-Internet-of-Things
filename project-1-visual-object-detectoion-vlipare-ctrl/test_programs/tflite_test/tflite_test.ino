#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"   // your .tflite model
#include "test_inputs.h"  // input1 and input2

// Global objects
tflite::MicroInterpreter* interpreter = nullptr;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;

uint8_t* tensorArena;
#define ARENA_SIZE (200*1024)

void setup() {
  tensorArena = (uint8_t*) heap_caps_malloc(ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensorArena) {
      Serial.println("PSRAM allocation failed!");
      while(1);
  }

  Serial.begin(115200);
  delay(2000);
  Serial.println("=== START ===");

  // Load model
  model = tflite::GetModel(saved_models_cifar_cnn_model_tflite);
  if (!model) {
    Serial.println("Failed to load model!");
    while (1);
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensorArena, ARENA_SIZE);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }

  Serial.println("Model initialized!");
}

void loop() {
    const int inputSize = 3072;
    const int outputSize = 10;

    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    // --- Test input1 ---
    memcpy(input->data.int8, input1, inputSize);

    unsigned long t1 = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long t2 = micros();

    if (invoke_status != kTfLiteOk) {
        Serial.println("Inference failed!");
        while (1);
    }

    Serial.print("Input1 raw output: ");
    for (int i = 0; i < outputSize; i++) {
        Serial.print(output->data.int8[i]);
        Serial.print(" ");
    }
    Serial.println();

    // Find predicted class (max value)
    int8_t* outData = output->data.int8;
    int predicted = 0;
    for (int i = 1; i < outputSize; i++) {
        if (outData[i] > outData[predicted]) predicted = i;
    }
    Serial.printf("Input1 predicted class: %d\n", predicted);
    Serial.printf("Inference time (us): %lu\n\n", t2 - t1);

    delay(1000);

    // --- Test input2 ---
    memcpy(input->data.int8, input2, inputSize);

    t1 = micros();
    invoke_status = interpreter->Invoke();
    t2 = micros();

    if (invoke_status != kTfLiteOk) {
        Serial.println("Inference failed!");
        while (1);
    }

    Serial.print("Input2 raw output: ");
    for (int i = 0; i < outputSize; i++) {
        Serial.print(output->data.int8[i]);
        Serial.print(" ");
    }
    Serial.println();

    // Find predicted class (max value)
    outData = output->data.int8;
    predicted = 0;
    for (int i = 1; i < outputSize; i++) {
        if (outData[i] > outData[predicted]) predicted = i;
    }
    Serial.printf("Input2 predicted class: %d\n", predicted);
    Serial.printf("Inference time (us): %lu\n\n", t2 - t1);

    while (1);
}