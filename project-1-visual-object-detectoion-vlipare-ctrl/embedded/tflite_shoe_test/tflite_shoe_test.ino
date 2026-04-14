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
#define ARENA_SIZE (600*1024)

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("=== START ===");

  tensorArena = (uint8_t*) heap_caps_malloc(ARENA_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensorArena) {
      Serial.println("PSRAM allocation failed!");
      while(1);
  }

  // Load model
  model = tflite::GetModel(shoe_detector_rgb_native_tflite);
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
    const int inputSize = 36864;
    const int outputSize = 1;

    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;

    Serial.print("Output scale: ");
    Serial.println(output_scale, 8);
    Serial.print("Output zero point: ");
    Serial.println(output_zero_point);

    // --- Test input1 ---
    memcpy(input->data.int8, input1, inputSize);

    unsigned long t1 = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long t2 = micros();

    if (invoke_status != kTfLiteOk) {
        Serial.println("Inference failed!");
        while (1);
    }

    int8_t raw_output1 = output->data.int8[0];
    float real_output1 = (raw_output1 - output->params.zero_point) * output->params.scale;

    // Prediction
    const char* predicted1 = (real_output1 >= 0.5f) ? "Shoe" : "NoShoe";

    // Print everything
    Serial.println("----- INPUT 1 -----");
    Serial.print("Real: ");
    Serial.println(input1_label);

    Serial.print("Predicted: ");
    Serial.println(predicted1);

    Serial.print("Confidence: ");
    Serial.println(real_output1, 6);

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

    int8_t raw_output2 = output->data.int8[0];
    float real_output2 = (raw_output2 - output->params.zero_point) * output->params.scale;

    // Prediction
    const char* predicted2 = (real_output2 >= 0.5f) ? "Shoe" : "NoShoe";

    // Print everything
    Serial.println("----- INPUT 2 -----");
    Serial.print("Real: ");
    Serial.println(input2_label);

    Serial.print("Predicted: ");
    Serial.println(predicted2);

    Serial.print("Confidence: ");
    Serial.println(real_output2, 6);

    Serial.printf("Inference time (us): %lu\n\n", t2 - t1);

    while (1);
}