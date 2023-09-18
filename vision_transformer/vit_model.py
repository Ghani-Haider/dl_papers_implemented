import torch
from torch import nn

# 1. create image patch embeddnigs (2D image linear projection layer) / Implementation of Equation 1
class Patch_Embed_Layer(nn.Module):
    def __init__(self, in_channels, num_patch, patch_size, embed_size) -> None:
        super().__init__()

        self.patch_embedding_layer = nn.Sequential(
            # extract the 2D feature maps (learnable patches)
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_patch,
                      kernel_size=patch_size,
                      stride=patch_size,
                      padding=0),
            # flatten the feature maps
            nn.Flatten(start_dim=2,
                       end_dim=3),
            #  linear projection to create patch embedings
            nn.Linear(in_features=14*14, out_features=embed_size)
        )
    
    def forward(self, x):
        # create patch embeddings for given images
        return self.patch_embedding_layer(x)
    
# 2. create MSA Block / Implementation of Equation 2 without skip connection
class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, embed_size, attn_dropout=0) -> None:
        super().__init__()

        # qkv for input in the attention block
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)

        # layer norm
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_size)
        
        # attention block
        self.self_attention_layer = nn.MultiheadAttention(embed_dim=embed_size,
                                                          num_heads=num_heads,
                                                          dropout=attn_dropout,
                                                          batch_first=True)
        
    def forward(self, x):

        x = self.layer_norm(x)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # get the attn output [0] and ignore attn output weights [1]
        return self.self_attention_layer(query=q,
                                         key=k,
                                         value=v,
                                         need_weights=False)[0]
    
# 3. MLP Block / Implementation of Equation 3 without skip connection
class MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 embed_size, # hidden size D from Table 1 ViT Paper
                 mlp_size:int=3072, # from Table 1 of ViT-Base
                 dropout:float=0.1): # from Table 3 of ViT-Base
        super().__init__()
        
        # layer normalization
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_size)

        # mlp layer as specified in ViT Paper
        self.mlp_layer = nn.Sequential(
            nn.Linear(embed_size, mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_size, embed_size),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.mlp_layer(self.layer_norm(x))

# 4. create Transformer encoder block
class EncoderBlock(nn.Module):
    def __init__(self, num_heads, embed_size, mlp_size, dropout_attn, dropout) -> None:
        super().__init__()

        # multi-headed attention block with layer norm
        self.msa_block = MultiheadAttention(num_heads,
                                            embed_size,
                                            dropout_attn)
        
        # mlp block with layer norm
        self.mlp_block = MultiLayerPerceptron(embed_size,
                                              mlp_size,
                                              dropout)
        
    def forward(self, x):
        # msa with layer norm + skip connection
        x = self.msa_block(x) + x
        # mlp with layer norm + skip connection
        return self.mlp_block(x) + x

# 5. complete ViT Architecture (ViT Base Model)
class ViTBase(nn.Module):
    def __init__(self, in_channels, num_patch, patch_size, embed_size, num_heads, num_layers, mlp_size, dropout_embeds, dropout_mlp, dropout_attn, out_features) -> None:
        super().__init__()
        
        # image patch embeddings
        self.patch_embedding = Patch_Embed_Layer(in_channels, num_patch, patch_size, embed_size)
        
        # class token embedding
        self.class_embedding = nn.Parameter(torch.randn(size=(1, 1, embed_size), requires_grad=True))
        
        # postition embeddings of flatten patches
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patch+1, embed_size), requires_grad=True))

        # embeddings dropout
        self.embedding_dropout = nn.Dropout(p=dropout_embeds)
        
        # encoder block layers
        self.encoder_blocks = nn.Sequential(
            *[EncoderBlock(num_heads, embed_size, mlp_size, dropout_attn, dropout_mlp) for _ in range(num_layers)]
        )

        # classification head for image classification
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_size),
            nn.Linear(embed_size, out_features)
        )

    def forward(self, x):
        batch_size = x.shape[0]  # get batch dimension

        # create the image patches' embeddings
        patch_embeddings = self.patch_embedding(x)
        
        # create the output tokens similar to BERT's class token for the batch
        class_embedding = self.class_embedding.expand(batch_size, -1, -1)

        # create the complete sequence of embeddings with class token, patch and positional embeddings
        x_embedded = torch.concat([class_embedding, patch_embeddings], dim=1) + self.position_embedding

        # run embedding dropout (Appendix B.1 in ViT Paper)
        x_embedded_drpout = self.embedding_dropout(x_embedded)
        
        # pass the input through the Encoder layers
        encoder_output = self.encoder_blocks(x_embedded_drpout)
        
        # get the class token embeddings
        class_token = encoder_output[:, 0,:]
        
        # get the output labels from class token
        return self.classifier_head(class_token)