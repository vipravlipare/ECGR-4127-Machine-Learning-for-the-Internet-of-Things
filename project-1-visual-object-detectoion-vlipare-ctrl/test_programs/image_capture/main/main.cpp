#include "esp_camera.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdio.h>

static const char *TAG = "image_capture";

extern "C" void app_main()
{
    ESP_LOGI(TAG, "Starting camera test");

    camera_config_t config = {};

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    config.pin_d0 = 11;
    config.pin_d1 = 9;
    config.pin_d2 = 8;
    config.pin_d3 = 10;
    config.pin_d4 = 12;
    config.pin_d5 = 18;
    config.pin_d6 = 17;
    config.pin_d7 = 16;

    config.pin_xclk = 15;
    config.pin_pclk = 13;
    config.pin_vsync = 6;
    config.pin_href = 7;

    config.pin_sccb_sda = 4;
    config.pin_sccb_scl = 5;

    config.pin_pwdn = -1;
    config.pin_reset = -1;

    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;

    esp_err_t err = esp_camera_init(&config);

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed");
        return;
    }

    ESP_LOGI(TAG, "Camera initialized");

    sensor_t *s = esp_camera_sensor_get();
    
    if (s) {
        s->set_brightness(s, 1);
        s->set_contrast(s, 1);
        s->set_saturation(s, 0);

        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);

        s->set_exposure_ctrl(s, 1);
        s->set_aec2(s, 0);
        s->set_gain_ctrl(s, 1);

        s->set_ae_level(s, 1);
        s->set_gainceiling(s, GAINCEILING_16X);

        s->set_lenc(s, 1);
        s->set_bpc(s, 1);
        s->set_wpc(s, 1);
    }

    while (1)
    {
        camera_fb_t *fb = esp_camera_fb_get();

        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        printf("IMAGE_START\n");

        for (size_t i = 0; i < fb->len; i++) {
            printf("%u,", (unsigned int)fb->buf[i]);
        }

        printf("\nIMAGE_END\n");

        esp_camera_fb_return(fb);

        vTaskDelay(pdMS_TO_TICKS(500));
    }
}