# 按键
## 1. 长短按
### key.c:
```#include "headfile.h"
#include "stm32g4xx_it.h"
void led_show(uint8_t led, uint8_t mode)
{
    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_2, GPIO_PIN_SET);

    // mode = 1 ???, mode = 0 ???
    if (mode)
        HAL_GPIO_WritePin(GPIOC, GPIO_PIN_8 << (led - 1), GPIO_PIN_RESET);  // ????
    else
        HAL_GPIO_WritePin(GPIOC, GPIO_PIN_8 << (led - 1), GPIO_PIN_SET);    // ????

    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_2, GPIO_PIN_RESET);
}
 
KEY_T key[4] = {0};   // 四个按键
uint8_t led_state[4] = {0};

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM4)  // 判断是否为TIM4的中断
    {
        // 读取按键状态
        key[0].key_sta = HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_0);
        key[1].key_sta = HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_1);
        key[2].key_sta = HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_2);
        key[3].key_sta = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0);

        // 遍历所有按键
        for (uint8_t i = 0; i < 4; i++)
        {
            switch (key[i].junk_sta)
            {
            case 0: // 无操作状态（未按下）
                if (key[i].key_sta == GPIO_PIN_RESET) // 检测到按下
                {
                    key[i].junk_sta = 1;
					keytime=0;
                }
                break;

            case 1: // 按下检测状态
                if (key[i].key_sta == GPIO_PIN_RESET)
                {
                    key[i].single_flag = 0;
                    key[i].long_flag = 0;				// 按下瞬间计时清零
                    key[i].junk_sta = 2;    // 进入按住状态
                }
                else
                {
                    key[i].junk_sta = 0;    // 抖动，返回未按下
                }
                break;

            case 2: 
                if (key[i].key_sta == GPIO_PIN_SET)
                {
					if (keytime >= 800)
                    {
                        key[i].long_flag = 1;  // 标记长按事件
                    }
					else if(keytime <= 800)
					{
						key[i].single_flag = 1;
					}
                    key[i].junk_sta = 0; // 回到初始状态
                }
                break;
            }
        }
    }
}
```
### key.h:
```
#ifndef _key_h
#define _key_h
#include "stdbool.h"
#include "stm32g4xx.h"  // Device header
void led_show(uint8_t led,uint8_t mode);

typedef struct
{
    GPIO_PinState key_sta;   // 当前按键状态
    uint8_t junk_sta;        // 消抖状态机状态
    uint8_t single_flag;     // 短按标志
    uint8_t long_flag;       // 长按标志
//	uint8_t mode_flag;
    uint32_t keytime;        // 按下持续时间（ms）
} KEY_T;

extern KEY_T key[4];
extern uint8_t led_state[4]; // ????LED?????(0=?,1=?)
 
#endif
```
### main.c
while(1)中的代码
- 示例：短按键1亮灯1，长按灯亮2
```
	 if (key[0].single_flag == 1)
    {
        key[0].single_flag = 0; // ???
        led_state[0] = !led_state[0]; // ????
        led_show(1, led_state[0]);
    }

    // ????2
    if (key[0].long_flag == 1)
    {
        key[0].long_flag = 0;
        led_state[1] = !led_state[1];
        led_show(2, led_state[1]);
    }
````
其他配置：在stm32g4xx_it.c中void SysTick_Handler(void)加入了  keytime++;  用于计时，需要在相应.h文件中进行声明变量keytime。
