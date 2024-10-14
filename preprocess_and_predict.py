from PIL import Image

def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((248, 248)),
        transforms.ToTensor(),
    ])

    return transform

def preprocess_and_predict(model, image_path, device):
    """
    Preprocess a single image and make a prediction using the trained model.

    Args:
        model: The trained CNN model.
        image_path (str): Path to the image file.
        device: The device to run the model on (CPU or GPU).

    Returns:
        predicted_class (int): The predicted class index.
        mean (float): The mean value of the image tensor.
        std (float): The standard deviation of the image tensor.
    """
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert image to grayscale

    # Transform the image
    transform = get_transforms()
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Move tensor to the specified device
    img_tensor = img_tensor.to(device)

    # Compute mean and std for the image
    mean = torch.mean(img_tensor)
    std = torch.std(img_tensor)

    # Make a prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(img_tensor, mean.unsqueeze(0).to(device), std.unsqueeze(0).to(device))  # Include mean and std in the prediction
        predicted_class = output.argmax(dim=1).item()  # Get the predicted class index

    return predicted_class, mean.item(), std.item()

# Example usage
image_path = '/kaggle/input/imagesoasis/Data/Very mild Dementia/OAS1_0003_MR1_mpr-1_102.jpg'
predicted_class_index, mean, std = preprocess_and_predict(traced_model, image_path, device)

print(f"Predicted Class Index: {predicted_class_index}, Mean: {mean}, Std: {std}")
if predicted_class_index == 0 :
    print ('Mild Dementia')
    
elif predicted_class_index ==1 :
    print('Moderate Dementia')
    
elif predicted_class_index == 2 :
    print ('Non Demented')
    
elif predicted_class_index ==3 :
    print('Very mild Dementia')
    
else :
    print('not supported')
