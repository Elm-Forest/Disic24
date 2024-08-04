import pandas as pd
import torch
from pytorch_widedeep import Trainer
from pytorch_widedeep.callbacks import LRHistory, EarlyStopping
from pytorch_widedeep.initializers import KaimingNormal
from pytorch_widedeep.metrics import Accuracy, Precision
from pytorch_widedeep.models import WideDeep, TabTransformer, Vision
from pytorch_widedeep.preprocessing import TabPreprocessor, ImagePreprocessor
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import ToTensor, Normalize, Resize

main_key = ['isic_id', 'target']

numeric_columns = [
    'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B',
    'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L',
    'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
    'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border',
    'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
    'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
]

categorical_columns = [
    'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location',
    'tbp_lv_location_simple', 'attribution'
]

def train(csv_file):
    df = pd.read_csv(csv_file, low_memory=False)
    df = df[numeric_columns + categorical_columns + main_key]
    for col in categorical_columns:
        df[col] = df[col].fillna('Unknown')

    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns].fillna(0))
    # Tabular
    tab_preprocessor = TabPreprocessor(
        embed_cols=categorical_columns, continuous_cols=numeric_columns, cols_to_scale=numeric_columns,
    )
    img_path = 'K:\\dataset\\Disic2024challenge\\train-image\\image'
    df['img_name'] = df['isic_id'] + '.jpg'
    image_processor = ImagePreprocessor(img_col='img_name', img_path=img_path)

    X_tab = tab_preprocessor.fit_transform(df)
    X_images = image_processor.fit_transform(df)
    # tab_mlp = TabMlp(
    #     column_idx=tab_preprocessor.column_idx,
    #     cat_embed_input=tab_preprocessor.cat_embed_input,
    #     continuous_cols=tab_preprocessor.continuous_cols,
    #     mlp_hidden_dims=[16, 8],
    # )
    # Pretrained Resnet 18
    resnet = Vision(pretrained_model_setup="resnet50", n_trainable=4)
    tab_transformer = TabTransformer(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        cat_embed_dropout=0.1,
        continuous_cols=numeric_columns,
        embed_continuous_method="standard",
        cont_norm_layer="layernorm",
        cont_embed_dropout=0.2,
        cont_embed_activation="leaky_relu",
        n_heads=4,
        ff_dropout=0.2,
        mlp_dropout=0.5,
        mlp_activation="leaky_relu",
        mlp_linear_first="True",
    )
    model = WideDeep(deeptabular=tab_transformer, deepimage=resnet, head_hidden_dims=[256, 128])
    deep_params = []
    for childname, child in model.named_children():
        if childname == "deeptabular":
            for n, p in child.named_parameters():
                if "embed_layer" in n:
                    deep_params.append({"params": p, "lr": 1e-4})
                else:
                    deep_params.append({"params": p, "lr": 1e-3})
    img_opt = torch.optim.AdamW(model.deepimage.parameters())
    deep_opt = torch.optim.Adam(deep_params)
    head_opt = torch.optim.Adam(model.deephead.parameters())
    deep_sch = torch.optim.lr_scheduler.MultiStepLR(deep_opt, milestones=[3, 8])
    img_sch = torch.optim.lr_scheduler.MultiStepLR(deep_opt, milestones=[3, 8])
    head_sch = torch.optim.lr_scheduler.StepLR(head_opt, step_size=5)
    # trainer = Trainer(model, objective="binary")
    optimizers = {
        "deeptabular": deep_opt,
        "deepimage": img_opt,
        "deephead": head_opt,
    }
    schedulers = {
        "deeptabular": deep_sch,
        "deepimage": img_sch,
        "deephead": head_sch,
    }
    initializers = {
        "deeptabular": KaimingNormal,
        "deepimage": KaimingNormal,
        "deephead": KaimingNormal,
    }

    mean = [0.406, 0.456, 0.485]  # BGR
    std = [0.225, 0.224, 0.229]  # BGR
    transforms = [ToTensor, Resize((128, 128)), Normalize(mean=mean, std=std)]
    callbacks = [
        LRHistory(n_epochs=10),
        EarlyStopping
    ]
    trainer = Trainer(
        model,
        objective="binary",
        initializers=initializers,
        optimizers=optimizers,
        lr_schedulers=schedulers,
        callbacks=callbacks,
        transforms=transforms,
    )
    # tab_trainer = Trainer(
    #     model=model,
    #     objective="binary",
    #     optimizers=torch.optim.AdamW(model.parameters(), lr=0.001),
    #     metrics=[Accuracy, Precision],
    # )
    trainer.fit(
        X_tab=X_tab,
        X_img=X_images,
        target=df["target"].values,
        n_epochs=10,
        batch_size=32,
        val_split=0.2,
        metrics=[Accuracy, Precision]
    )
    # Option 1: this will also save training history and lr history if the
    # LRHistory callback is used
    trainer.save(path="./model_weights.pt", save_state_dict=True)

    # Option 2: save as any other torch model
    torch.save(model.state_dict(), "./wd_model.pt")

if __name__ == "__main__":
    csv_file = 'K:/dataset/Disic2024challenge/train-metadata.csv'
    csv_file = 'data_folder/balanced_dataset.csv'
    train(csv_file)
