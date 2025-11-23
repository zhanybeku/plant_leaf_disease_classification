#include <leaf_binary_inferencing.h>

#define FRAME_SIZE (96*96)
uint8_t frame[FRAME_SIZE];

int get_data_from_camera(size_t offset, size_t length, float *out_ptr) {
  for(size_t i = 0; i < length; ++i) {
    uint8_t gray = frame[i + offset];
    uint8_t r = gray;
    uint8_t g = gray;
    uint8_t b = gray;

    float pixel_f = (r << 16) + (g << 8) + b;
    out_ptr[i] = pixel_f;
  }
  return 0;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
}

void loop() {
  Serial.println("<cam-read>");
  Serial.readBytes(frame, FRAME_SIZE);

  ei::signal_t signal;
  signal.total_length = FRAME_SIZE;
  signal.get_data = &get_data_from_camera;

  ei_impulse_result_t result = { 0 };

  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    Serial.print("ERROR: ");
    Serial.println(err);
    return;
  }

  Serial.println("Classification results:");
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    Serial.print("  ");
    Serial.print(result.classification[ix].label);
    Serial.print(": ");
    Serial.println(result.classification[ix].value, 5);
  }

  float best_val = 0.0;
  const char* best_label = "";

  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    if (result.classification[ix].value > best_val) {
      best_val = result.classification[ix].value;
      best_label = result.classification[ix].label;
    }
  }

  Serial.print("BEST: ");
  Serial.print(best_label);
  Serial.print(" (");
  Serial.print(best_val, 3);
  Serial.println(")");

  delay(50);
}