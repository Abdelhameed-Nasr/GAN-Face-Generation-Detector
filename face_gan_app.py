"""
Face-GAN App -- Generate Faces & Detect AI-Generated Images
=============================================================

A standalone app (no Jupyter/notebook needed). Run it with:

    python face_gan_app.py

This opens a page in your browser with two tabs:
  1. Generate Faces  -- makes new synthetic faces from your trained Generator
  2. Detect AI Image -- checks whether an uploaded image "looks real" or
                        "looks AI-generated" according to your Discriminator

IMPORTANT HONESTY NOTE on the detector: it was trained to catch fakes from
THIS specific Generator, trained on THIS specific dataset. It is NOT a
general-purpose AI-image detector -- it will not reliably catch images from
other tools (Midjourney, DALL-E, other GANs, etc.), and may be wrong on real
photos that look statistically different from the training data. Treat its
output as "does this look like what my Generator produces", not a
trustworthy forensic verdict on arbitrary images.

SETUP (one-time):
    pip install torch torchvision pillow gradio

Then put your downloaded .pth checkpoint files in a folder called
"checkpoints" next to this script (or edit CHECKPOINT_FOLDER below), and run:

    python face_gan_app.py
"""

import os
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ==============================================================================
# 1. CONFIG -- edit these to match your setup
# ==============================================================================

ARCHITECTURE = "old"        # "old" (feat=64, no attention) or "new" (feat=96, self-attention + minibatch-stddev)
CHECKPOINT_EPOCH = 255       # match this to an epoch you actually have a checkpoint for
CHECKPOINT_FOLDER = "checkpoints"   # folder next to this script with your downloaded .pth files

IMG_SIZE = 128
LATENT_DIM = 128
CHANNELS = 3

if ARCHITECTURE == "old":
    GEN_FEAT, DISC_FEAT = 64, 64
    USE_ATTENTION, USE_MINIBATCH_STDDEV = False, False
elif ARCHITECTURE == "new":
    GEN_FEAT, DISC_FEAT = 96, 96
    USE_ATTENTION, USE_MINIBATCH_STDDEV = True, True
else:
    raise ValueError('ARCHITECTURE must be "old" or "new"')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 2. MODEL DEFINITIONS -- must match the training notebook exactly
#
# IMPORTANT: the "old" and "new" architectures are NOT the same class with a
# toggle -- they have genuinely different internal layer structure (the old
# one is a single flat nn.Sequential named "net"; the new one is split into
# "block1"/"attention"/"block2"). Loading a checkpoint into the wrong
# structure fails with a "Missing/Unexpected key(s)" error, because the
# saved parameter names literally don't match. So each gets its own class.
# ==============================================================================

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, max(channels // 8, 1), 1)
        self.key = nn.Conv2d(channels, max(channels // 8, 1), 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class MinibatchStdDev(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)
        mean_std = std.mean().expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, mean_std], dim=1)


# ---- OLD architecture (pre-upgrade -- flat "net" Sequential, no attention) ----

class GeneratorOld(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, feat=GEN_FEAT, channels=CHANNELS, img_size=IMG_SIZE):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(latent_dim, feat * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat * 8), nn.ReLU(True),
            nn.ConvTranspose2d(feat * 8, feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 4), nn.ReLU(True),
            nn.ConvTranspose2d(feat * 4, feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 2), nn.ReLU(True),
            nn.ConvTranspose2d(feat * 2, feat, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat), nn.ReLU(True),
            nn.ConvTranspose2d(feat, feat // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat // 2), nn.ReLU(True),
            nn.ConvTranspose2d(feat // 2, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        return self.net(z)


class DiscriminatorOld(nn.Module):
    def __init__(self, feat=DISC_FEAT, channels=CHANNELS, img_size=IMG_SIZE, dropout=0.1):
        super().__init__()
        layers = [
            nn.Conv2d(channels, feat // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat // 2, feat, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(feat, feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 2), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(feat * 2, feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 4), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(feat * 4, feat * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 8, 1, 4, 1, 0, bias=False),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


# ---- NEW architecture (self-attention + minibatch-stddev upgrade) ----

class GeneratorNew(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, feat=GEN_FEAT, channels=CHANNELS, img_size=IMG_SIZE):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feat * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat * 8), nn.ReLU(True),
            nn.ConvTranspose2d(feat * 8, feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 4), nn.ReLU(True),
            nn.ConvTranspose2d(feat * 4, feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 2), nn.ReLU(True),
            nn.ConvTranspose2d(feat * 2, feat, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat), nn.ReLU(True),
        )
        self.attention = SelfAttention(feat)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(feat, feat // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat // 2), nn.ReLU(True),
            nn.ConvTranspose2d(feat // 2, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        x = self.block1(z)
        x = self.attention(x)
        return self.block2(x)


class DiscriminatorNew(nn.Module):
    def __init__(self, feat=DISC_FEAT, channels=CHANNELS, img_size=IMG_SIZE, dropout=0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(channels, feat // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat // 2, feat, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
        )
        self.attention = SelfAttention(feat)
        block2 = [
            nn.Conv2d(feat, feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 2), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(feat * 2, feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 4), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(feat * 4, feat * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 8), nn.LeakyReLU(0.2, inplace=True),
            MinibatchStdDev(),
            nn.Conv2d(feat * 8 + 1, 1, 4, 1, 0, bias=False),
        ]
        self.block2 = nn.Sequential(*block2)

    def forward(self, x):
        x = self.block1(x)
        x = self.attention(x)
        x = self.block2(x)
        return x.view(-1)


Generator = GeneratorOld if ARCHITECTURE == "old" else GeneratorNew
Discriminator = DiscriminatorOld if ARCHITECTURE == "old" else DiscriminatorNew


# ==============================================================================
# 3. LOAD THE TRAINED CHECKPOINT
# ==============================================================================

def find_checkpoints(folder, epoch):
    """Recursively searches `folder` for a matching generator/generator_ema/
    discriminator triple. Falls back to the closest available epoch if the
    exact one isn't found."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"'{folder}' doesn't exist. Create that folder and put your downloaded "
            f".pth checkpoint files in it, or edit CHECKPOINT_FOLDER at the top of this script."
        )
    pattern = re.compile(r"(generator_ema|generator|discriminator)_gan_epoch_(\d+)\.pth$")
    by_kind_epoch = {"generator": {}, "discriminator": {}, "generator_ema": {}}
    for dirpath, _dirs, filenames in os.walk(folder):
        for fn in filenames:
            m = pattern.match(fn)
            if m:
                kind, ep = m.group(1), int(m.group(2))
                by_kind_epoch[kind][ep] = os.path.join(dirpath, fn)

    common_epochs = sorted(set(by_kind_epoch["generator"]) & set(by_kind_epoch["discriminator"]))
    if not common_epochs:
        raise FileNotFoundError(
            f"No generator/discriminator checkpoint pairs found under '{folder}'. "
            f"Filenames should look like 'generator_gan_epoch_255.pth' and "
            f"'discriminator_gan_epoch_255.pth'."
        )

    chosen = epoch if epoch in common_epochs else common_epochs[-1]
    if chosen != epoch:
        print(f"Epoch {epoch} not found -- found {common_epochs}, using {chosen} instead.")

    return (by_kind_epoch["generator"][chosen], by_kind_epoch["discriminator"][chosen],
            by_kind_epoch["generator_ema"].get(chosen), chosen)


print(f"Using device: {DEVICE}")
print(f"Architecture: {ARCHITECTURE} (gen_feat={GEN_FEAT}, disc_feat={DISC_FEAT}, "
      f"attention={USE_ATTENTION}, minibatch_stddev={USE_MINIBATCH_STDDEV})")

gen_path, disc_path, ema_path, CHECKPOINT_EPOCH = find_checkpoints(CHECKPOINT_FOLDER, CHECKPOINT_EPOCH)
print(f"Generator:     {gen_path}")
print(f"Discriminator: {disc_path}")
print(f"EMA generator: {ema_path if ema_path else '(not found -- using raw generator instead)'}")

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)
generator.load_state_dict(torch.load(gen_path, map_location=DEVICE))
discriminator.load_state_dict(torch.load(disc_path, map_location=DEVICE))
if ema_path:
    generator.load_state_dict(torch.load(ema_path, map_location=DEVICE))  # prefer EMA weights for quality

generator.eval()
discriminator.eval()
print(f"Loaded checkpoint from epoch {CHECKPOINT_EPOCH}. Ready.\n")


# ==============================================================================
# 4. APP LOGIC
# ==============================================================================

resize_crop_preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

exact_size_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def preprocess(image):
    """Uses the resize+crop pipeline for arbitrary-sized real photos (standardizes
    framing), but SKIPS it for images already exactly IMG_SIZE x IMG_SIZE (e.g. a
    face this same app just generated) -- resizing an already-correct-size image
    still applies interpolation that subtly shifts pixel values, which is why a
    generated face could score differently in "Detect" than it did in "Generate"
    if it went through an unnecessary resize round-trip."""
    if image.size == (IMG_SIZE, IMG_SIZE):
        return exact_size_preprocess(image)
    return resize_crop_preprocess(image)


def tensor_to_pil(tensor):
    """Converts a single [-1,1]-normalized CHW tensor back to a displayable PIL image."""
    img = ((tensor + 1) / 2).clamp(0, 1)
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(img)


def generate_faces(n_faces):
    n_faces = int(n_faces)
    with torch.no_grad():
        z = torch.randn(n_faces, LATENT_DIM, device=DEVICE)
        faces = generator(z).detach().cpu()
    return [tensor_to_pil(faces[i]) for i in range(n_faces)]


def classify_image(image):
    if image is None:
        return "Upload an image first."
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = discriminator(tensor)
        prob_real = torch.sigmoid(logit).item()

    verdict = "Looks REAL" if prob_real > 0.5 else "Looks AI-GENERATED"
    confidence = max(prob_real, 1 - prob_real)

    return (
        f"**{verdict}**  ({confidence:.1%} confidence)\n\n"
        f"P(real) = {prob_real:.3f}\n\n"
        f"---\n"
        f"⚠️ Reminder: this only reflects similarity to THIS Generator's outputs, "
        f"not a general AI-detection verdict. It will not reliably catch images "
        f"from other AI tools, and may misjudge real photos that differ from the training data."
    )


# ==============================================================================
# 5. GRADIO UI (PREMIUM DARK SAPPHIRE & GOLD THEME)
# ==============================================================================

import gradio as gr

premium_gold_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_width="1px",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    button_primary_background_fill="#d4af37",
    button_primary_background_fill_dark="#d4af37",
    button_primary_text_color="#0f172a",
    button_primary_text_color_dark="#0f172a",
    button_secondary_background_fill="#1e293b",
    button_secondary_background_fill_dark="#1e293b",
    button_secondary_text_color="#f8fafc",
    button_secondary_text_color_dark="#f8fafc",
    body_text_color="#f8fafc",
    body_text_color_dark="#f8fafc",
    block_title_text_color="#d4af37",
    block_title_text_color_dark="#d4af37",
    block_label_background_fill="#1e293b",
    block_label_background_fill_dark="#1e293b",
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    border_color_primary="#334155",
    code_background_fill="#1e293b",
    code_background_fill_dark="#1e293b",
)

css = """
@keyframes sapphireGlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

body, .gradio-container {
    background: linear-gradient(-45deg, #0f172a, #1e3a8a, #0369a1, #0f172a) !important;
    background-size: 400% 400% !important;
    animation: sapphireGlow 15s ease infinite !important;
    font-family: 'Outfit', sans-serif !important;
    color: #f8fafc !important;
}

.container { max-width: 1000px; margin: auto; padding-top: 30px; }

/* Shiny Gold Title */
h1 {
    color: transparent !important;
    background: linear-gradient(135deg, #FFF8D6 0%, #D4AF37 50%, #996515 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    text-shadow: 0px 4px 15px rgba(212, 175, 55, 0.4);
    font-weight: 900 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.markdown-text, .gr-markdown p, .gr-markdown h2, .gr-markdown h3 { color: #e0f2fe !important; }

/* Shiny Gold Button */
button.primary {
    background: linear-gradient(135deg, #FFF8D6 0%, #D4AF37 50%, #996515 100%) !important;
    background-size: 200% auto !important;
    color: #0f172a !important;
    border: none !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4) !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    background-position: right center !important;
    box-shadow: 0 8px 25px rgba(212, 175, 55, 0.6) !important;
    transform: translateY(-2px) !important;
}

code { 
    background-color: #1e293b !important; 
    color: #d4af37 !important; 
    border-radius: 6px;
}

/* Tabs styling */
.tab-nav button {
    color: #94a3b8 !important;
    font-weight: 600 !important;
}
.tab-nav button.selected {
    color: #d4af37 !important;
    border-bottom-color: #d4af37 !important;
}

/* Fix Gallery and Image White/Light Background Boxes */
.gradio-container [data-testid="gallery"],
.gradio-container [data-testid="image"],
.gradio-container .gallery,
.gradio-container .image-container,
.gradio-container .image-frame,
.gradio-container .frame,
.gradio-container .preview,
.gradio-container .preview-image,
.gradio-container .thumbnails,
.gradio-container .thumbnail-item,
.gradio-container button.gallery-item,
.gradio-container div[class*="preview"],
.gradio-container div[class*="gallery"],
.gradio-container div[class*="image"],
.gradio-container div[class*="bg-white"],
.gradio-container div[class*="bg-gray"] {
    background: #1e293b !important;
    background-color: #1e293b !important;
    border-color: #334155 !important;
    color: #f8fafc !important;
}

/* Dark canvas behind the actual displayed image and thumbnails */
.gradio-container [data-testid="gallery"] > div,
.gradio-container [data-testid="image"] > div,
.gradio-container .preview > div,
.gradio-container img {
    background: #0f172a !important;
    background-color: #0f172a !important;
}

/* Selected gallery thumbnail gold glow */
.gradio-container [data-testid="gallery"] button.selected,
.gradio-container button.gallery-item.selected {
    border-color: #d4af37 !important;
    box-shadow: 0 0 12px rgba(212, 175, 55, 0.6) !important;
}
"""

with gr.Blocks(title="Face-GAN App", theme=premium_gold_theme, css=css) as demo:
    gr.Markdown("# 🎭 Face-GAN App")
    gr.Markdown(
        "Generate new synthetic faces, or check whether an image looks real or AI-generated "
        "(according to this specific trained model)."
    )

    with gr.Tab("Generate Faces"):
        n_faces_input = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="Number of faces")
        generate_btn = gr.Button("Generate", variant="primary")
        faces_gallery = gr.Gallery(label="Generated faces", columns=4, height="auto")
        generate_btn.click(fn=generate_faces, inputs=n_faces_input, outputs=faces_gallery)

    with gr.Tab("Detect AI Image"):
        gr.Markdown(
            "⚠️ **Honesty note:** this only checks similarity to THIS model's own fakes, "
            "not general AI-image detection. See the top of this app for details."
        )
        image_input = gr.Image(type="pil", label="Upload an image")
        classify_btn = gr.Button("Classify", variant="primary")
        result_output = gr.Markdown()
        classify_btn.click(fn=classify_image, inputs=image_input, outputs=result_output)

if __name__ == "__main__":
    demo.launch()
