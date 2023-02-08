_base_ = './QDTrack_PCAN_fixed.py'
# model settings
model = dict(
    type='EMQuasiDenseMaskRCNNRefine',
    roi_head=dict(
        type='QuasiDenseSegRoIHeadRefine',
        double_train=False,
        mask_head=dict(type='FCNMaskHeadPlus'),
        refine_head=dict(
            type='EMMatchHeadPlus',
            num_convs=4,
            in_channels=256,
            conv_kernel_size=3,
            conv_out_channels=256,
            upsample_method='deconv',
            upsample_ratio=2,
            num_classes=24,
            pos_proto_num=30,  # ori 10           # saturate at 30 (said in the PCAN paper)
            neg_proto_num=30,  # ori 10
            stage_num=6,
            conv_cfg=None,
            norm_cfg=None,
            mask_thr_binary=0.5,
            match_score_thr=0.5,
            with_mask_ref=False,
            with_mask_key=True,
            with_dilation=False,
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0))),
    tracker=dict(type='QuasiDenseSegFeatEmbedTracker'))



dataset_type = 'SAILVOSVideoDatasetNoJoint'
data_root = '/beegfs/work/shared/kangdong_shared/sailvos_cut_png/'            # SAIL-VOScut
# data_root = '/beegfs/work/shared/kangdong_shared/sailvos_complete_video/'   # SAIL-VOS
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=[
        dict(
            type='SAILVOSVideoDatasetNoJoint',


            ## for AmodalPCAN training on SAIL-VOScut:
            ann_file="/beegfs/work/shared/kangdong_shared/sailvos_cut_json/png_amodal/train_less0.75_png_amodal.json",

            ## for PCAN training on SAIL-VOScut:
            # ann_file="/beegfs/work/shared/kangdong_shared/sailvos_cut_json/png_visible/train_less0.75_png_visible.json",

            img_prefix='/beegfs/work/shared/kangdong_shared/sailvos_cut_png/',



            ## for AmodalPCAN training on SAIL-VOS:
            # ann_file="/beegfs/work/shared/kangdong_shared/sailvos_complete_json/png_amodal_cmplt_video/train_less0.75_png_amodal_cmplt_vid.json",

            ## for PCAN training on SAIL-VOS:
            # ann_file="/beegfs/work/shared/kangdong_shared/sailvos_complete_json/png_visible_cmplt_video/train_less0.75_png_visible_cmplt_vid.json",

            # img_prefix='/beegfs/work/shared/kangdong_shared/sailvos_complete_video/',


            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(num_ref_imgs=1, scope=3, method='uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations',
                    with_bbox=True,
                    with_ins_id=True,
                    with_mask=True),
                dict(type='SeqResize', img_scale=(1280, 800), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=32),
                dict(type='SeqDefaultFormatBundle'),
                dict(
                    type='SeqCollect',
                    keys=[
                        'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
                        'gt_masks'
                    ],
                    ref_prefix='ref')
            ])
    ],
    val=dict(
        type='SAILVOSVideoDatasetNoJoint',


        ## for AmodalPCAN validation on SAIL-VOScut:
        ann_file="/beegfs/work/shared/kangdong_shared/sailvos_cut_json/png_amodal/valid_less0.75_png_amodal.json",

        ## for PCAN validation on SAIL-VOScut:
        # ann_file="/beegfs/work/shared/kangdong_shared/sailvos_cut_json/png_visible/valid_less0.75_png_visible.json",

        img_prefix='/beegfs/work/shared/kangdong_shared/sailvos_cut_png/',



        ## for AmodalPCAN validation on SAIL-VOS:
        # ann_file="/beegfs/work/shared/kangdong_shared/sailvos_complete_json/png_amodal_cmplt_video/valid_less0.75_png_amodal_cmplt_vid.json",

        ## for PCAN validation on SAIL-VOS:
        # ann_file="/beegfs/work/shared/kangdong_shared/sailvos_complete_json/png_visible_cmplt_video/valid_less0.75_png_visible_cmplt_vid.json",

        # img_prefix='/beegfs/work/shared/kangdong_shared/sailvos_complete_video/',


        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ]),
    test=dict(
        type='SAILVOSVideoDatasetNoJoint',


        ## for AmodalPCAN testing on SAIL-VOScut:
        ann_file="/beegfs/work/shared/kangdong_shared/sailvos_cut_json/png_amodal/valid_less0.75_png_amodal.json",

        ## for PCAN testing on SAIL-VOScut:
        # ann_file="/beegfs/work/shared/kangdong_shared/sailvos_cut_json/png_visible/valid_less0.75_png_visible.json",

        img_prefix='/beegfs/work/shared/kangdong_shared/sailvos_cut_png/',



        ## for AmodalPCAN testing on SAIL-VOS:
        # ann_file="/beegfs/work/shared/kangdong_shared/sailvos_complete_json/png_amodal_cmplt_video/valid_less0.75_png_amodal_cmplt_vid.json",

        ## for PCAN testing on SAIL-VOS:
        # ann_file="/beegfs/work/shared/kangdong_shared/sailvos_complete_json/png_visible_cmplt_video/valid_less0.75_png_visible_cmplt_vid.json",

        # img_prefix='/beegfs/work/shared/kangdong_shared/sailvos_complete_video/',


        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ]))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=1)
total_epochs = 12
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 2000,
    step=[8, 11])


load_from = '/beegfs/work/kangdongjin/pcan/train_files/1007_kidl/epoch_10.pth'  # trained QDTrack_mots weights file
resume_from = None