import os
import click
from melo.api import TTS
from nemo_text_processing.text_normalization.normalize import Normalizer

normalizer = Normalizer(input_case='cased', lang='en')
    
    
# @click.command()
# @click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the checkpoint file")
# @click.option('--text', '-t', type=str, default=None, help="Text to speak")
# @click.option('--language', '-l', type=str, default="EN", help="Language of the model")
# @click.option('--output_dir', '-o', type=str, default="outputs", help="Path to the output")
def inference(ckpt_path, text, number, language = 'EN', output_dir = 'outputs'):
    if ckpt_path is None:
        raise ValueError("The model_path must be specified")
    
    config_path = 'logs/example/config.json'
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)
    
    for spk_name, spk_id in model.hps.data.spk2id.items():
        save_path = f'{output_dir}/{spk_name}/{number}.wav'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.tts_to_file(text, spk_id, save_path)

if __name__ == "__main__":
    ckpt_path = 'logs/example/G_202000.pth'
    texts = [
    "Welcome to Verizon! Enjoy blazing-fast internet speeds with our 5G Home Internet.",
    "Experience the future with Verizon's ultra-reliable 5G network.",
    "Setting up your 5G Home Internet is simple and quick. Get started today!",
    "Our latest offer includes a free month of service for new customers.",
    "Say goodbye to buffering with Verizon 5G Home Internet.",
    "The time now is 10:30 AM. We'll begin in 30 minutes.",
    "Your appointment is scheduled for January 15, 2023, at 2:00 PM.",
    "The event starts at 7:00 PM on October 10, 2023.",
    "The deadline for the project is March 3, 2023.",
    "The meeting is scheduled for February 28, 2023.",
    "Visit our website at https://www.verizon.com for more information.",
    "Contact us at support@verizon.com for assistance.",
    "Send your feedback to feedback@verizon.com.",
    "The total cost is $49.99 per month.",
    "The price of the item is $19.99.",
    "The current exchange rate is 1 USD to 0.85 EUR.",
    "Please call us at (123) 456-7890 for more details.",
    "Plug in the 5G router, follow the simple instructions, and you'll be online in no time.",
    "Don't miss out on the future of home internet.",
    "Do you have any questions about our 5G Home Internet?",
    "Yes, our customer service is here to support you.",
    "Our office is located at 1234 Main St, Anytown, USA.",
    "Visit us at 5678 Market St, Suite 101, Big City, USA.",
    "Even if you have multiple devices connected simultaneously, Verizon 5G Home Internet ensures smooth performance.",
    "With Verizon's advanced network, you can enjoy lag-free gaming, uninterrupted video calls, and seamless streaming, all at once.",
    "Our 5G network not only provides high speed but also offers exceptional reliability, ensuring you stay connected during peak hours.",
    "Verizon 5G adapts to your needs, whether you're streaming, gaming, or working.",
    "Enjoy high-speed internet with Verizon 5G Home Internet.",
    "Join the future of home internet with Verizon 5G.",
    "Stay connected to what matters most with Verizon.",
    "Upgrade your home internet with Verizon 5G.",
    "Explore the potential of smart home devices with Verizon's reliable network.",
    "With our flexible plans, you can choose the one that best suits your household needs.",
    "Experience uninterrupted entertainment with Verizon 5G's superior bandwidth.",
    "Connect effortlessly to your virtual meetings with Verizon's stable internet service.",
    "Enhance your work-from-home setup with Verizon 5G's top-notch performance.",
    "The time right now is 10:30 AM. We will start working in 30 minutes.",
    "No need to wait for a technician—just plug in the 5G router, follow the simple instructions, and you'll be online in no time.",
    "Inquiry texts Sure thing. Got it. Understood. Okay. I see. Alright. Noted. Interesting. Gotcha. Right.",
    "Complaint Sorry about that. I see. Got it. Understood. I apologize. I hear you. That's frustrating. My bad. We’re sorry. I get your point. I understand. I'm sorry to hear that.",
    "Feedback Thank you. Got it. Noted. I see. Understood. Okay. Right. Appreciate it. Thanks. Good to know. Thanks for it.",
    "Confirmation Got it. Understood. Okay. Noted. Right. Sure. Yes. Absolutely. Exactly. For sure. Definitely. You bet. yeah. Alright.",
    "Gratitude You're welcome. Happy to help. No problem. Anytime. Glad to help. My pleasure. No worries. Of course. Sure thing. No issue.",
    "Apology No problem. It's okay. Don't worry. All good. That's fine. No worries. No issue. Understandable.",
    "Request Please hold on. One moment, please. Let me assist. Let me check. Let me see Give me a moment Give me a second Okay. Ofcourse Certainly. ",
    "Disagreement I understand. I see your point. That's valid. I get it. I hear you. Ok, Let's discuss I understand your point. Let's talk more."
    ]
    
    for i, text in enumerate(texts):
        text = normalizer.normalize(text)
        inference(ckpt_path, text, number = i)
