
#include <TensorFlowLite.h>
#include <TinyMLShield.h>
//#include <Arduino_OV767X.h>

#include <string.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

#include "model.h"

// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* modelVb = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
  constexpr int kTensorArenaSize = 136 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}


unsigned short pixels[176 * 144]; //QCIF

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("OV767X Camera Capture");
  Serial.println();

  if (!Camera.begin(QCIF, RGB565, 1, OV7675)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  modelVb = tflite::GetModel(model);

  if (modelVb->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         modelVb->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only needed operations (should match NN layers)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  //static tflite::MicroMutableOpResolver<1> micro_mutable_op_resolver;
  //tflite_status = micro_op_resolver.AddFullyConnected();
  //micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED(), 1, 3);

  // Pull in only needed operations (should match NN layers)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  /*static tflite::MicroMutableOpResolver<1> micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_FULLY_CONNECTED,
    tflite::ops::micro::Register_FULLY_CONNECTED(),
    1, 3);*/
    //tflite_status = micro_op_resolver.AddFullyConnected();
  
  // Build an interpreter to run the model
  
  static tflite::MicroInterpreter static_interpreter(
    modelVb, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  Serial.println("Send the 'g' character to read a frame ...");
  Serial.println();

}

void loop() {
  if (Serial.read() == 'g') {
    Serial.println("Reading frame");
    Serial.println();
    Camera.readFrame(pixels);

    int numPixels = Camera.width() * Camera.height();

    for (int i = 0; i < numPixels; i++) {
        unsigned short p = pixels[i];
        Serial.print("0x");
        if (p < 0x1000) {
          Serial.print('0');
        }
        if (p < 0x0100) {
         Serial.print('0');
        }
        if (p < 0x0010) {
          Serial.print('0');
        }
        Serial.print(p, HEX);
        Serial.print(", ");
      }
  }
}
