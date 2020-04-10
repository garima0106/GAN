from deep_privacy.config_parser import load_config
from deep_privacy.inference import infer, deep_privacy_anonymizer
import torch

config = load_config("models/large/config.yml")
checkpoint = torch.load("models/large/checkpoints/step_40000000.ckpt")
generator = infer.init_generator(config, checkpoint)

anonymizer = deep_privacy_anonymizer.DeepPrivacyAnonymizer(generator,
                                                           batch_size=32,
                                                           use_static_z=True,
                                                           keypoint_threshold=.1,
                                                           face_threshold=.6)
#anonymizer.anonymize_image_paths(["images/demo.jpg"], ["images/demo_anonymized.jpg"])
anonymizer.anonymize_video(["images/portrait.mp4"], ["images/portrait_anon.mp4"])
