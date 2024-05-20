from pathlib import Path

import orjson
from tqdm import tqdm
from imgutils.tagging import get_wd14_tags

from common import safe_name, txt2img, txt2img_nai3, check_model, 参数相同, 模型数据, 要测的人


check_model(模型数据)


存图文件夹 = Path('out_人')

sampler = 'DPM++ 2M Karras'
seed = 1
steps = 30
width = 512
height = 768
cfg_scale = 7


def 评测模型(model, VAE) -> list[dict]:
    存档文件名 = f'savedata/人物_{model}_记录.json'
    if Path(存档文件名).exists():
        with open(存档文件名, 'r', encoding='utf8') as f:
            记录 = orjson.loads(f.read())
    else:
        记录 = []
    for index, 人 in enumerate(tqdm(要测的人, ncols=70, desc=model[:10])):
        人 = 人.strip().replace(' ', '_')
        参数 = {
            'prompt': f'1 girl, {人}',
            'negative_prompt': 'worst quality, low quality',
            'seed': seed,
            'width': width,
            'height': height,
            'steps': steps,
            'sampler_index': sampler,
            'cfg_scale': cfg_scale,
            'override_settings': {
                'sd_model_checkpoint': model,
                'sd_vae': VAE,
                'CLIP_stop_at_last_layers': 1,
            },
        }
        skip = False
        for i in 记录:
            if i['人'] == 人 and 参数相同(i['参数'], 参数):
                skip = True
                break
        if skip:
            continue
        数量参数 = {
            'batch_size': 4,
            'n_iter': 2,
        }
        if model == 'nai-diffusion-3':
            图s = txt2img_nai3(数量参数 | 参数)
        else:
            图s = txt2img(数量参数 | 参数)
        for i, b in enumerate(图s):
            with open(存图文件夹 / safe_name(f'{人}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png'), 'wb') as f:
                f.write(b)
        n = len(图s)
        预测 = [get_wd14_tags(存图文件夹 / safe_name(f'{人}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png'), character_threshold=0.5)[2] for i in range(n)]
        # 预测 = [waifu_sensor.v1.predict(Image.open()), size=512, top_n=5) for i in range(n)]
        录 = {
            '预测': 预测,
            '总数': n,
            '人': 人,
            '参数': 参数,
        }
        记录.append(录)
        if True:
            with open(存档文件名, 'wb') as f:
                f.write(orjson.dumps(记录, default=float))
    return 记录


for model, VAE, *_ in tqdm(模型数据, ncols=70, desc='all'):
    评测模型(model, VAE)
