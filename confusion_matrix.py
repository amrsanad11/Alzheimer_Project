predictions, labels = predict(model, test_loader, criterion, device, eval=False)
conf_mat = confusion_matrix(labels, predictions)
class_to_idx = list(train_loader.dataset.subset.dataset.class_to_idx)
df_cm = pd.DataFrame(conf_mat, index = class_to_idx, columns = class_to_idx)
sn.heatmap(df_cm, annot=True, fmt='', cmap='Blues')
