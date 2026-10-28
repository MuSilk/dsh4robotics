# RoboTwin 可选任务

`server_robotwin.py --task` 接受下面 50 个任务名，也可以填 `all`。默认是 `adjust_bottle`。

```bash
python3 server_robotwin.py --task beat_block_hammer --gpu 0
```

单轮上限是该任务的官方 step limit。不写 `--horizon` 时用这一列。

| 任务 | 在做什么 | 单轮上限 |
|---|---|---|
| `adjust_bottle` | 用正确的手臂把瓶子头朝上拿起来 | 400 |
| `beat_block_hammer` | 拿起锤子敲方块 | 400 |
| `blocks_ranking_rgb` | 把红、绿、蓝方块从左到右排成一排 | 1200 |
| `blocks_ranking_size` | 把三个方块从大到小、从左到右排好 | 1200 |
| `click_alarmclock` | 按闹钟顶部中间的按钮 | 400 |
| `click_bell` | 按铃铛顶部中心 | 400 |
| `dump_bin_bigbin` | 拿起小盒，把球倒进大盒 | 600 |
| `grab_roller` | 双手抓住滚筒 | 400 |
| `handover_block` | 左手抓红块，交给右手，放到蓝垫上 | 800 |
| `handover_mic` | 一只手抓麦克风，交给另一只手 | 600 |
| `hanging_mug` | 拿起杯子转正，再挂到架子上 | 900 |
| `lift_pot` | 双手把锅抬起来 | 400 |
| `move_can_pot` | 把罐子拿到锅旁边 | 400 |
| `move_pillbottle_pad` | 把药瓶放到垫子上 | 400 |
| `move_playingcard_away` | 把扑克牌从桌面上挪走 | 400 |
| `move_stapler_pad` | 把订书机移到彩色垫上 | 400 |
| `open_laptop` | 打开笔记本电脑 | 700 |
| `open_microwave` | 打开微波炉 | 1500 |
| `pick_diverse_bottles` | 两只手各拿一个不同的瓶子 | 400 |
| `pick_dual_bottles` | 两只手各拿一个瓶子 | 400 |
| `place_a2b_left` | 把物体 A 放到物体 B 的左边 | 400 |
| `place_a2b_right` | 把物体 A 放到物体 B 的右边 | 400 |
| `place_bread_basket` | 把桌上的面包都放进篮子 | 700 |
| `place_bread_skillet` | 一只手托平底锅，另一只手把面包放进去 | 500 |
| `place_burger_fries` | 双手同时拿起汉堡和薯条，放到托盘上 | 500 |
| `place_can_basket` | 把罐子放进篮子，再用另一只手提起篮子 | 700 |
| `place_cans_plasticbox` | 双手各拿一个罐子，放进塑料盒 | 800 |
| `place_container_plate` | 把容器放到盘子上 | 400 |
| `place_dual_shoes` | 双手把两只鞋放进鞋盒，鞋尖朝左 | 600 |
| `place_empty_cup` | 把空杯子放到杯垫上 | 500 |
| `place_fan` | 把风扇放到彩色垫上，风扇朝向机器人 | 400 |
| `place_mouse_pad` | 把鼠标放到彩色垫上 | 400 |
| `place_object_basket` | 把目标放进篮子，再拿起篮子稍微挪开 | 700 |
| `place_object_scale` | 把物体放到秤上 | 400 |
| `place_object_stand` | 把物体放到支架上 | 400 |
| `place_phone_stand` | 把手机放到手机支架上 | 400 |
| `place_shoe` | 把鞋放到垫子上 | 500 |
| `press_stapler` | 按订书机 | 400 |
| `put_bottles_dustbin` | 把瓶子放进桌子左侧的垃圾桶 | 1700 |
| `put_object_cabinet` | 一只手拿物体，另一只手开抽屉，再放进去 | 700 |
| `rotate_qrcode` | 拿起二维码板转正，再放回桌上 | 400 |
| `scan_object` | 一只手拿扫描器，另一只手拿物体，完成扫描 | 500 |
| `shake_bottle` | 用合适的手臂摇瓶子 | 700 |
| `shake_bottle_horizontally` | 水平摇瓶子 | 700 |
| `stack_blocks_three` | 在中间叠三个方块：红在下，绿在中，蓝在上 | 1200 |
| `stack_blocks_two` | 把绿方块叠到红方块上 | 800 |
| `stack_bowls_three` | 把三个碗叠在一起 | 1200 |
| `stack_bowls_two` | 把两个碗叠在一起 | 900 |
| `stamp_seal` | 拿起印章，盖到指定颜色的垫子上 | 400 |
| `turn_switch` | 拨开关 | 400 |

任务名来自 `third_party/RoboTwin/env_cfg/eval/all_tasks.yml`，单轮上限来自 `third_party/RoboTwin/env_cfg/task_config/_eval_step_limit.yml`。
