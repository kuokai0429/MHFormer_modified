# 2023.0519 @Brian

import torch
import torch.nn as nn
from einops import rearrange
from model.module.trans import Transformer_Paper as Transformer_encoder_Paper
from model.module.trans import Transformer_Proposed_1 as Transformer_encoder_Proposed_1
from model.module.trans_hypothesis import Transformer_Paper as Transformer_hypothesis_Paper
from model.module.trans_hypothesis import Transformer_Proposed_1 as Transformer_hypothesis_Proposed_1

class Model_Paper(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## Normalization @Paper
        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        self.norm_3 = nn.LayerNorm(args.frames)

        ## MHG [B(JC)F] @Paper
        self.Transformer_encoder_1 = Transformer_encoder_Paper(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder_Paper(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder_Paper(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)

        ## MHG [B(JC)F] RealFormer using ReLA @Propopsed
        # self.Transformer_encoder_1 = Transformer_encoder_Proposed_1(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        # self.Transformer_encoder_2 = Transformer_encoder_Proposed_1(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        # self.Transformer_encoder_3 = Transformer_encoder_Proposed_1(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)

        ## Embedding @Paper
        if args.frames > 27:
            self.embedding_1 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(2*args.out_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR [BF(JC)] + CHI [BJ(JC)] @Paper
        self.Transformer_hypothesis = Transformer_hypothesis_Paper(args.layers, args.channel, args.d_hid, length=args.frames)
        
        ## Regression @Paper
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args.channel*3, momentum=0.1),
            nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
        )
        

    # MHG[B(JC)F] + SHR[BF(JC)] + CHI[BF(JC)] @Paper
    def forward(self, x):
        '''
            0 torch.Size([256, 81, 17, 2])
            1 torch.Size([256, 34, 81])
            2 torch.Size([256, 34, 81])
            3 torch.Size([256, 81, 512])
            4 torch.Size([256, 81, 1536])
            5 torch.Size([256, 1536, 81])
            6 torch.Size([256, 51, 81])
            7 torch.Size([256, 81, 17, 3])
        '''

        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        ## MHG : b (j c) f
        x_1 = x   + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1)) 
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))

        ## Embedding : b (j c) f -> b f (j c) -> b f (j c)
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous() 
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()

        ## SHR (Sequence coherence) & CHI : b f (j c)
        x = self.Transformer_hypothesis(x_1, x_2, x_3) 

        ## Regression : b f (j c) -> b (j c) f -> b f j c
        x = x.permute(0, 2, 1).contiguous()
        x = self.regression(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x


class Model_Proposed_1(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## Normalization @Paper
        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        self.norm_3 = nn.LayerNorm(args.frames)

        ## MHG [B(JC)F] @Paper
        self.Transformer_encoder_1 = Transformer_encoder_Paper(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder_Paper(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder_Paper(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)

        ## MHG [B(JC)F] MHG [B(JC)F] RealFormer using ReLA @Propopsed
        # self.Transformer_encoder_1 = Transformer_encoder_Proposed_1(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        # self.Transformer_encoder_2 = Transformer_encoder_Proposed_1(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        # self.Transformer_encoder_3 = Transformer_encoder_Proposed_1(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)

        ## Embedding @Paper
        if args.frames > 27:
            self.embedding_1 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(2*args.out_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR [BF(JC)] + CHI [BJ(JC)] @Paper
        self.Transformer_hypothesis_2 = Transformer_hypothesis_Paper(args.layers, args.channel, args.d_hid, length=args.frames)
        
        ## Regression @Paper
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args.channel*3, momentum=0.1),
            nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
        )

        ## 2023.0521 SHR [(BF)JC] @Brian
        self.Transformer_hypothesis_1 = Transformer_hypothesis_Proposed_1(args.layers, 32, args.d_hid, length=args.frames)

        ## 2023.0521 Embedding & Regression @Brian
        self.frames = args.frames
        if args.frames > 27:
            self.Spatial_Patch_1 = nn.Conv1d(2, 32, kernel_size=1)
            self.Spatial_Patch_2 = nn.Conv1d(2, 32, kernel_size=1)
            self.Spatial_Patch_3 = nn.Conv1d(2, 32, kernel_size=1)
        else:
            self.Spatial_Patch_1 = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=1),
                nn.BatchNorm1d(32, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )
            self.Spatial_Patch_2 = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=1),
                nn.BatchNorm1d(32, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )
            self.Spatial_Patch_3 = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=1),
                nn.BatchNorm1d(32, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )
        self.Spatial_Patch_4 = nn.Sequential(
            nn.Conv1d(args.n_joints*32, args.channel, kernel_size=1)
        )

    # 2023.0521 MHG[B(JC)F] + SHR[(BF)JC] + SHR[BF(JC)] + CHI[BF(JC)] @Brian
    def forward(self, x):
        '''
            0 torch.Size([256, 81, 17, 2])
            1 torch.Size([256, 34, 81])
            2 torch.Size([256, 34, 81])
            3 torch.Size([256, 81, 17, 2])
            4 torch.Size([20736, 17, 2])
            5 torch.Size([20736, 17, 32])
            6 torch.Size([20736, 17, 32])
            7 torch.Size([256, 81, 17, 32])
            8 torch.Size([256, 544, 81])
            9 torch.Size([256, 81, 512])
            10 torch.Size([256, 81, 1536])
            11 torch.Size([256, 1536, 81])
            12 torch.Size([256, 51, 81])
            13 torch.Size([256, 81, 17, 3])
        '''

        # print(f"0 {x.shape}")
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()


        ### MHG : b (j c) f

        # print(f"1 {x.shape}")
        x_1 = x   + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1)) 
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))


        ### Embedding : b (j c) f -> (b f) j c @Brian

        # print(f"2 {x_1.shape}")
        x_1 = rearrange(x_1, 'b (j c) f -> b f j c', j=J).contiguous()
        x_2 = rearrange(x_2, 'b (j c) f -> b f j c', j=J).contiguous()
        x_3 = rearrange(x_3, 'b (j c) f -> b f j c', j=J).contiguous()
        # print(f"3 {x_1.shape}")
        x_1 = rearrange(x_1, 'b f j c -> (b f) j c').contiguous()
        x_2 = rearrange(x_2, 'b f j c -> (b f) j c').contiguous()
        x_3 = rearrange(x_3, 'b f j c -> (b f) j c').contiguous()
        # print(f"4 {x_1.shape}")
        x_1 = self.Spatial_Patch_1(x_1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x_2 = self.Spatial_Patch_2(x_2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x_3 = self.Spatial_Patch_3(x_3.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()


        ### SHR (Kinematic correlation) : (b f) j c @Brian

        # print(f"5 {x_1.shape}")
        x_1, x_2, x_3 = self.Transformer_hypothesis_1(x_1, x_2, x_3)
        # print(f"6 {x_1.shape}")
        x_1 = rearrange(x_1, '(b f) j c -> b f j c', f=self.frames).contiguous()
        x_2 = rearrange(x_2, '(b f) j c -> b f j c', f=self.frames).contiguous()
        x_3 = rearrange(x_3, '(b f) j c -> b f j c', f=self.frames).contiguous()
        # print(f"7 {x_1.shape}")
        x_1 = rearrange(x_1, 'b f j c -> b (j c) f').contiguous()
        x_2 = rearrange(x_2, 'b f j c -> b (j c) f').contiguous()
        x_3 = rearrange(x_3, 'b f j c -> b (j c) f').contiguous()

        
        ### Embedding : b (j c) f -> b f (j c) @Brian

        # print(f"8 {x_1.shape}")
        x_1 = self.Spatial_Patch_4(x_1).permute(0, 2, 1).contiguous() 
        x_2 = self.Spatial_Patch_4(x_2).permute(0, 2, 1).contiguous() 
        x_3 = self.Spatial_Patch_4(x_3).permute(0, 2, 1).contiguous() 


        ### SHR (Sequence coherence) & CHI : b f (j c)

        # print(f"9 {x_1.shape}")
        x = self.Transformer_hypothesis_2(x_1, x_2, x_3) 


        ### Regression : b f (j c) -> b (j c) f -> b f j c

        # print(f"10 {x.shape}")
        x = x.permute(0, 2, 1).contiguous()
        # print(f"11 {x.shape}")
        x = self.regression(x)
        # print(f"12 {x.shape}")
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()
        # print(f"13 {x.shape}")

        return x

class Model_Proposed_2(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## Normalization @Paper
        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        self.norm_3 = nn.LayerNorm(args.frames)

        ## MHG [B(JC)F] @Paper
        self.Transformer_encoder_1 = Transformer_encoder_Paper(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder_Paper(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder_Paper(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)

        ## Embedding @Paper
        if args.frames > 27:
            self.embedding_1 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(2*args.out_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR [BF(JC)] + CHI [BJ(JC)] @Paper
        self.Transformer_hypothesis = Transformer_hypothesis_Paper(args.layers, args.channel, args.d_hid, length=args.frames)
        
        ## Regression @Paper
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args.channel*3, momentum=0.1),
            nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
        )

    # 2023.0521 MHG[B(JC)F] + SHR[(BF)JC] + SHR[BF(JC)] + CHI[BF(JC)] @Brian
    def forward(self, x):
        '''
            0 torch.Size([256, 81, 17, 2])
            1 torch.Size([256, 34, 81])
            2 torch.Size([256, 34, 81])
            3 torch.Size([256, 81, 17, 2])
            4 torch.Size([20736, 17, 2])
            5 torch.Size([20736, 17, 32])
            6 torch.Size([20736, 17, 32])
            7 torch.Size([256, 81, 17, 32])
            8 torch.Size([256, 544, 81])
            9 torch.Size([256, 81, 512])
            10 torch.Size([256, 81, 1536])
            11 torch.Size([256, 1536, 81])
            12 torch.Size([256, 51, 81])
            13 torch.Size([256, 81, 17, 3])
        '''

        # print(f"0 {x.shape}")
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()


        ### MHG : b (j c) f

        # print(f"1 {x.shape}")
        x_1 = x   + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1)) 
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))


        ### Embedding : b (j c) f -> (b f) j c @Brian

        # print(f"2 {x_1.shape}")
        x_1 = rearrange(x_1, 'b (j c) f -> b f j c', j=J).contiguous()
        x_2 = rearrange(x_2, 'b (j c) f -> b f j c', j=J).contiguous()
        x_3 = rearrange(x_3, 'b (j c) f -> b f j c', j=J).contiguous()
        # print(f"3 {x_1.shape}")
        x_1 = rearrange(x_1, 'b f j c -> (b f) j c').contiguous()
        x_2 = rearrange(x_2, 'b f j c -> (b f) j c').contiguous()
        x_3 = rearrange(x_3, 'b f j c -> (b f) j c').contiguous()
        # print(f"4 {x_1.shape}")
        x_1 = self.Spatial_Patch_1(x_1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x_2 = self.Spatial_Patch_2(x_2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x_3 = self.Spatial_Patch_3(x_3.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()


        ### SHR (Kinematic correlation) : (b f) j c @Brian

        # print(f"5 {x_1.shape}")
        x_1, x_2, x_3 = self.Transformer_hypothesis_1(x_1, x_2, x_3)
        # print(f"6 {x_1.shape}")
        x_1 = rearrange(x_1, '(b f) j c -> b f j c', f=self.frames).contiguous()
        x_2 = rearrange(x_2, '(b f) j c -> b f j c', f=self.frames).contiguous()
        x_3 = rearrange(x_3, '(b f) j c -> b f j c', f=self.frames).contiguous()
        # print(f"7 {x_1.shape}")
        x_1 = rearrange(x_1, 'b f j c -> b (j c) f').contiguous()
        x_2 = rearrange(x_2, 'b f j c -> b (j c) f').contiguous()
        x_3 = rearrange(x_3, 'b f j c -> b (j c) f').contiguous()

        
        ### Embedding : b (j c) f -> b f (j c) @Brian

        # print(f"8 {x_1.shape}")
        x_1 = self.Spatial_Patch_4(x_1).permute(0, 2, 1).contiguous() 
        x_2 = self.Spatial_Patch_4(x_2).permute(0, 2, 1).contiguous() 
        x_3 = self.Spatial_Patch_4(x_3).permute(0, 2, 1).contiguous() 


        ### SHR (Sequence coherence) & CHI : b f (j c)

        # print(f"9 {x_1.shape}")
        x = self.Transformer_hypothesis_2(x_1, x_2, x_3) 


        ### Regression : b f (j c) -> b (j c) f -> b f j c

        # print(f"10 {x.shape}")
        x = x.permute(0, 2, 1).contiguous()
        # print(f"11 {x.shape}")
        x = self.regression(x)
        # print(f"12 {x.shape}")
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()
        # print(f"13 {x.shape}")

        return x
