import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import gradio as gr


def load_clip_model(model_path, config):
    """Load a trained CLIP model from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIP(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def prepare_image(image):
    """Prepare image for CLIP model inference."""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return transform(image).unsqueeze(0)


def prepare_text(text_list):
    """Prepare text prompts for CLIP model inference."""
    tokens_list = []
    masks_list = []
    
    for text in text_list:
        tokens, mask = tokenizer(text)
        tokens_list.append(tokens)
        masks_list.append(mask)
    
    return torch.stack(tokens_list), torch.stack(masks_list)


def get_image_text_similarity(model, image_tensor, text_tokens, text_masks):
    """Calculate similarity between image and text prompts."""
    with torch.no_grad():
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        text_tokens = text_tokens.to(device)
        text_masks = text_masks.to(device)
        
        image_features = model.image_encoder(image_tensor)
        text_features = model.text_encoder(text_tokens, mask=text_masks)
        
        similarities = (image_features @ text_features.transpose(-2, -1)).squeeze(0)
        similarities = similarities * torch.exp(model.temperature)
        return torch.nn.functional.softmax(similarities, dim=-1).cpu().numpy()


def create_html_results(prompts, scores):
    """Create HTML output for displaying results."""
    best_idx = np.argmax(scores)
    sorted_indices = np.argsort(scores)[::-1]
    
    html = """
    <div style="max-width: 800px; margin: 0 auto; font-family: sans-serif;">
        <div style="background: linear-gradient(to right, #4287f5, #42a5f5); 
                    color: white; 
                    padding: 20px; 
                    border-radius: 10px 10px 0 0;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h2 style="margin: 0; font-size: 1.5em;">Best Match</h2>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">
                "{}"
            </p>
            <div style="margin-top: 10px; font-size: 1.1em;">
                Confidence: <strong>{:.1f}%</strong>
            </div>
        </div>
    """.format(prompts[best_idx], scores[best_idx]*100)
    
    html += """
        <div style="background: white; 
                    border-radius: 0 0 10px 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    overflow: hidden;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #f8f9fa;">
                    <th style="padding: 15px; text-align: left; color: #2c3e50;">Prompt</th>
                    <th style="padding: 15px; text-align: right; color: #2c3e50;">Score</th>
                </tr>
    """
    
    for idx in sorted_indices:
        score_percentage = scores[idx] * 100
        background = f"background: linear-gradient(to right, rgba(66, 135, 245, 0.1) {score_percentage}%, white {score_percentage}%)"
        html += """
                <tr style="{}">
                    <td style="padding: 12px 15px; border-top: 1px solid #eee;">
                        {}
                    </td>
                    <td style="padding: 12px 15px; text-align: right; border-top: 1px solid #eee;">
                        <strong>{:.1f}%</strong>
                    </td>
                </tr>
        """.format(background, prompts[idx], score_percentage)
    
    html += """
            </table>
        </div>
    </div>
    """
    
    return html


def analyze_image(image, text_input):
    """Gradio interface function for image analysis."""
    if image is None:
        return gr.HTML("""
            <div style="padding: 20px; 
                        background: #ffe6e6; 
                        border-radius: 10px; 
                        color: #cc0000;
                        text-align: center;
                        margin: 20px 0;">
                <h3 style="margin: 0;">⚠️ Please upload an image to analyze</h3>
            </div>
        """)
    
    text_prompts = [x.strip() for x in text_input.split('\n') if x.strip()]
    
    if not text_prompts:
        return gr.HTML("""
            <div style="padding: 20px; 
                        background: #ffe6e6; 
                        border-radius: 10px; 
                        color: #cc0000;
                        text-align: center;
                        margin: 20px 0;">
                <h3 style="margin: 0;">⚠️ Please enter at least one text prompt</h3>
            </div>
        """)
    
    image_tensor = prepare_image(image)
    text_tokens, text_masks = prepare_text(text_prompts)
    similarities = get_image_text_similarity(model, image_tensor, text_tokens, text_masks)
    
    return create_html_results(text_prompts, similarities)


# Model configuration
config = {
    'emb_dim': 512,
    'width': 768,
    'img_size': (224, 224),
    'patch_size': (16, 16),
    'n_channels': 3,
    'vit_layers': 6,
    'vit_heads': 12,
    'vocab_size': 50000,
    'text_width': 512,
    'max_seq_length': 77,
    'text_heads': 8,
    'text_layers': 6
}

# Load model 
MODEL_PATH = 'clip.pt'
model = load_clip_model(MODEL_PATH, config)

# Create Gradio interface
demo = gr.Blocks(theme=gr.themes.Soft())

with demo:
    gr.Markdown("""
    # CLIP Understanding: Image to Text
    Upload an image and provide multiple text descriptions to see how well they match. 
    Enter each potential text prompt on a new line, and our CLIP model will rank them by similarity to your image.
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="numpy",
                label="Upload Image"
            )
            text_input = gr.Textbox(
                lines=5,
                placeholder="Enter text prompts (one per line)",
                label="Text Prompts"
            )
            analyze_button = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            results_output = gr.HTML(label="Results")
    
    analyze_button.click(
        fn=analyze_image,
        inputs=[image_input, text_input],
        outputs=results_output
    )
    
    gr.Examples(
        examples=[
            [
                "image.jpg",
                "a dog playing in the park\n" +
                "a girl is standing with horse near fire\n" +
                "a cat sleeping on a couch\n" +
                "a bird flying in the sky"
            ]
        ],
        inputs=[image_input, text_input]
    )

if __name__ == "__main__":
    demo.launch(share=True)

