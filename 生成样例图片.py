from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm

from common import 模型数据
from backend_diffusers import txt2img


模型数据d = {i[0]: i for i in 模型数据}

要测的模型 = ['sweetfruit_melon.safetensors_v1.0', 'coharumix_v6', 'cuteyukimixAdorable_specialchapter', 'koji_v21', 'rabbit_v7', 'irismix_v90', 'jitq_v30',  'kaywaii_v70', 'Counterfeit-V3.0_fp16', 'darkSushiMixMix_225D', 'bluePencil_v10', 'rimochan_random_mix_4.0','cetusMix_v4', 'etherBluMix_etherBluMix5', 'meinamix_meinaV11', 'mixProV4_v4', 'AOM3A1', 'Aidv210AnimeIllustDiffusion_aidv210', 'himawarimix_v100', 'ghostmix_v20Bakedvae', 'calicomix_v75', 'novelailatest-pruned', 'aiceKawaice_channel', 'AnythingV5Ink_ink', 'divineelegancemix_V9', 'anyloraCheckpoint_novaeFp16', 'counterfeitxl_v25', 'netaArtXL_v10', 'ConfusionXL4.0_fp16_vae', 'flux1-schnell-fp8']


存图文件夹 = Path('out_样例图片')
存图文件夹.mkdir(exist_ok=True)


prompts = [
    'scenary, nature, outdoors, tree, river, no humans',
    'fantasy, 6+girls, outdoors',
    '1girl, twintails, school uniform, upper body, looking at viewer, outdoors, street',
    '1girl, witch, test tube, indoors, upper body, desk',
    '1girl, long hair, coffee, cafe, collared shirt, cowboy shot, indoors',
    '1girl, twintails, long hair, night, gothic dress, looking at viewer, reclining, car interior',
    '1girl, twintails, cat ears, maid, maid headdress, holding tray, white pantyhose, indoors, kitchen',
    '1girl, twintails, samurai, japanese armor, seiza, indoors, cup',
    '1girl, scientist, sitting, chair, blonde hair, twintails, steampunk, gears, cowboy shot, indoors',
]
width = 512
height = 768


for model in tqdm(要测的模型):
    VAE = 模型数据d[model][1]
    model_type = 模型数据d[model][3]
    for prompt in prompts:
        if Path(存图文件夹 / f'{model}×{VAE}×{prompt}.png').exists():
            continue
        参数 = {
            'prompt': prompt,
            'negative_prompt': '(worst quality, low quality:1.2)',
            'seed': 1,
            'width': width,
            'height': height,
            'steps': 50,
            'sampler_name': 'DPM++ 2M',
            'scheduler': 'Karras',
            'cfg_scale': 7,
            'override_settings': {
                'sd_model_checkpoint': model,
                'sd_vae': VAE,
                'CLIP_stop_at_last_layers': 1,
            },
            'model_type': model_type,
        }
        数量参数 = {
            'batch_size': 1,
            'n_iter': 1,
        }
        r = txt2img(数量参数 | 参数)
        with open(存图文件夹 / f'{model}×{VAE}×{prompt}.png', 'wb') as f:
            f.write(r[0])

for prompt in prompts:
    COLUMN_MAX = 7
    ROW_MAX = (len(要测的模型) - 1) // COLUMN_MAX + 1
    x = 0
    y = 0
    target = Image.new('RGB', (width*COLUMN_MAX, (height+100)*ROW_MAX), color=(255, 255, 255))
    for model in tqdm(要测的模型):
        VAE = 模型数据d[model][1]
        image = Image.open(存图文件夹 / f'{model}×{VAE}×{prompt}.png')
        dr = ImageDraw.Draw(target)
        if len(model) <= 24:
            name = model
        else:
            name = model[:21] + '...'
        dr.text((x * width, y * (height+100)+50), name, font=ImageFont.truetype('consola.ttf', 38), fill=(0, 0, 0))
        target.paste(image, (x * width, y * (height+100) + 100, (x + 1) * width, (y + 1) * (height+100)))
        if x == COLUMN_MAX - 1:
            x = 0
            y += 1
        else:
            x += 1
    target.save(存图文件夹 / f'__final_{prompt}.png')
    target = target.resize((target.size[0] // 2, target.size[1] // 2), Image.Resampling.LANCZOS)
    target.save(存图文件夹 / f'_final_{prompt}.webp', quality=75)
