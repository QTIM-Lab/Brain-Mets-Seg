import tensorflow as tf

from models import layers
import numpy as np

#NOTE: UNet     = Trainable params: 31,194,017, with filters: [32, 64, 128, 256, 320, 320]
#NOTE: UNet_Jay = Trainable params: 31,112,161, with filters: [32, 64, 128, 256, 320, 320]
#NOTE: UNet implements (conv-norm-act)x2 in each layer, with the first conv having stride of 2 if needing to downsample in the encoder. For the decoder, a 2x2x2 filter convtranspose is used for upsampling (and it reduces filters to the number in the next level up so that the skip connection branch and upsampled feature maps have the same number of channels). Scale and center is set to true for instance norm, and all biases on convolutions are turned on.
#NOTE: My implementation of U-Net reduces features via a kernel size 1 filter before the upsampling/concatenation operation (as opposed to upsampling/concatenating first and then reducing dimensionality (or not reducing dimensionality at all))

class UNet(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        n_class,
        kernels,
        strides,
        normalization_layer,
        negative_slope,
        dimension,
        deep_supervision,
        max_pool=False,
    ):
        super().__init__()
        self.dim = dimension
        self.n_class = n_class
        self.negative_slope = negative_slope
        self.norm = normalization_layer
        self.deep_supervision = deep_supervision
        filters = [min(2 ** (5 + i), 320 if dimension == 3 else 512) for i in range(len(strides))]
        #filters = [48,96,160,240,336,528]
        self.filters = filters
        self.kernels = kernels
        self.strides = strides

        down_block = layers.ConvBlock
        self.input_block = self.get_conv_block(
            conv_block=down_block,
            filters=filters[0],
            kernel_size=kernels[0],
            stride=strides[0],
            input_shape=input_shape,
        )
        self.downsamples = self.get_block_list(
            conv_block=down_block, filters=filters[1:], kernels=kernels[1:-1], strides=strides[1:-1]
        )
        self.bottleneck = self.get_conv_block(
            conv_block=down_block, filters=filters[-1], kernel_size=kernels[-1], stride=strides[-1]
        )
        self.upsamples = self.get_block_list(
            conv_block=layers.UpsampleBlock,
            filters=filters[:-1][::-1],
            kernels=kernels[1:][::-1],
            strides=strides[1:][::-1],
        )
        self.output_block = self.get_output_block()
        if self.deep_supervision:
            self.deep_supervision_heads = [self.get_output_block(), self.get_output_block()]
        self.n_layers = len(self.upsamples) - 1

    def call(self, x, training=True):
        skip_connections = []
        out = self.input_block(x)
        skip_connections.append(out)

        for down_block in self.downsamples:
            out = down_block(out)
            skip_connections.append(out)

        out = self.bottleneck(out)

        decoder_outputs = []
        for up_block in self.upsamples:
            out = up_block(out, skip_connections.pop())
            decoder_outputs.append(out)

        out = self.output_block(out)

        if training and self.deep_supervision:
            out = [
                out,
                self.deep_supervision_heads[0](decoder_outputs[-2]),
                self.deep_supervision_heads[1](decoder_outputs[-3]),
            ]
        return out

    def get_output_block(self):
        return layers.OutputBlock(filters=self.n_class, dim=self.dim, negative_slope=self.negative_slope)

    def get_conv_block(self, conv_block, filters, kernel_size, stride, **kwargs):
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            kernel_size=kernel_size,
            filters=filters,
            negative_slope=self.negative_slope,
            **kwargs,
        )

    def get_block_list(self, conv_block, filters, kernels, strides):
        layers = []
        for filter, kernel, stride in zip(filters, kernels, strides):
            conv_layer = self.get_conv_block(conv_block, filter, kernel, stride)
            layers.append(conv_layer)
        return layers


class UNet_Jay(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        n_class,
        kernels,
        strides,
        normalization_layer,
        negative_slope,
        dimension,
        deep_supervision,
        max_pool=True,
    ):
        super().__init__()
        self.dim = dimension
        self.n_class = n_class
        self.negative_slope = negative_slope
        self.norm = normalization_layer
        self.deep_supervision = deep_supervision
        self.kernels = kernels
        self.strides = strides
        self.filter_num_per_level = [min(2 ** (5 + i), 320 if dimension == 3 else 512) for i in range(len(self.strides))]
        self.filters = self.filter_num_per_level
        #
        #encoder layers
        self.encoder_layer_1 = layers.conv_layer_encoder(num_filters=self.filter_num_per_level[0], use_norm_act=False, use_pooling=False, negative_slope=self.negative_slope, norm=self.norm, input_shape=input_shape)
        self.encoder_layer_2 = layers.conv_layer_encoder(num_filters=self.filter_num_per_level[1], use_pooling=False, negative_slope=self.negative_slope, norm=self.norm)
        self.encoder_layer_3 = layers.conv_layer_encoder(num_filters=self.filter_num_per_level[2], use_pooling=False, negative_slope=self.negative_slope, norm=self.norm)
        self.encoder_layer_4 = layers.conv_layer_encoder(num_filters=self.filter_num_per_level[3], use_pooling=False, negative_slope=self.negative_slope, norm=self.norm)
        self.encoder_layer_5 = layers.conv_layer_encoder(num_filters=self.filter_num_per_level[4], use_pooling=False, negative_slope=self.negative_slope, norm=self.norm)
        #
        #pooling layers
        if max_pool == True:
            self.pooling_layer_1 = tf.keras.layers.MaxPool3D(pool_size=2)
            self.pooling_layer_2 = tf.keras.layers.MaxPool3D(pool_size=2)
            self.pooling_layer_3 = tf.keras.layers.MaxPool3D(pool_size=2)
            self.pooling_layer_4 = tf.keras.layers.MaxPool3D(pool_size=2)
            self.pooling_layer_5 = tf.keras.layers.MaxPool3D(pool_size=2)
        else:
            self.pooling_layer_1 = layers.conv_downsampling(num_filters=self.filter_num_per_level[0], prior_filter_num=self.filter_num_per_level[0], negative_slope=self.negative_slope, norm=self.norm)
            self.pooling_layer_2 = layers.conv_downsampling(num_filters=self.filter_num_per_level[1], prior_filter_num=self.filter_num_per_level[1], negative_slope=self.negative_slope, norm=self.norm)
            self.pooling_layer_3 = layers.conv_downsampling(num_filters=self.filter_num_per_level[2], prior_filter_num=self.filter_num_per_level[2], negative_slope=self.negative_slope, norm=self.norm)
            self.pooling_layer_4 = layers.conv_downsampling(num_filters=self.filter_num_per_level[3], prior_filter_num=self.filter_num_per_level[3], negative_slope=self.negative_slope, norm=self.norm)
            self.pooling_layer_5 = layers.conv_downsampling(num_filters=self.filter_num_per_level[4], prior_filter_num=self.filter_num_per_level[4], negative_slope=self.negative_slope, norm=self.norm)
        #
        #bottleneck
        self.bottleneck = layers.conv_layer_decoder(current_level_num_feature=self.filter_num_per_level[5], next_level_num_features=self.filter_num_per_level[4], reduce_features=True, use_upsample=False, negative_slope=self.negative_slope, norm=self.norm)
        #
        #decoder
        self.upsampling_layer_5 = layers.conv_upsampling(num_filters=self.filter_num_per_level[4], prior_filter_num=self.filter_num_per_level[4], negative_slope=self.negative_slope, norm=self.norm)
        self.decoder_layer_5 = layers.conv_layer_decoder(current_level_num_feature=self.filter_num_per_level[4], next_level_num_features=self.filter_num_per_level[3], reduce_features=True, use_upsample=False, negative_slope=self.negative_slope, norm=self.norm)
        self.upsampling_layer_4 = layers.conv_upsampling(num_filters=self.filter_num_per_level[3], prior_filter_num=self.filter_num_per_level[3], negative_slope=self.negative_slope, norm=self.norm)
        self.decoder_layer_4 = layers.conv_layer_decoder(current_level_num_feature=self.filter_num_per_level[3], next_level_num_features=self.filter_num_per_level[2], reduce_features=True, use_upsample=False, negative_slope=self.negative_slope, norm=self.norm)
        self.upsampling_layer_3 = layers.conv_upsampling(num_filters=self.filter_num_per_level[2], prior_filter_num=self.filter_num_per_level[2], negative_slope=self.negative_slope, norm=self.norm)
        self.decoder_layer_3 = layers.conv_layer_decoder(current_level_num_feature=self.filter_num_per_level[2], next_level_num_features=self.filter_num_per_level[1], reduce_features=True, use_upsample=False, negative_slope=self.negative_slope, norm=self.norm)
        self.upsampling_layer_2 = layers.conv_upsampling(num_filters=self.filter_num_per_level[1], prior_filter_num=self.filter_num_per_level[1], negative_slope=self.negative_slope, norm=self.norm)
        self.decoder_layer_2 = layers.conv_layer_decoder(current_level_num_feature=self.filter_num_per_level[1], next_level_num_features=self.filter_num_per_level[0], reduce_features=True, use_upsample=False, negative_slope=self.negative_slope, norm=self.norm)
        self.upsampling_layer_1 = layers.conv_upsampling(num_filters=self.filter_num_per_level[0], prior_filter_num=self.filter_num_per_level[0], negative_slope=self.negative_slope, norm=self.norm)
        self.decoder_layer_1 = layers.conv_layer_decoder(current_level_num_feature=self.filter_num_per_level[0], next_level_num_features=self.filter_num_per_level[0], reduce_features=False, use_upsample=False, negative_slope=self.negative_slope, norm=self.norm)
        #
        #output layers / deep supervision
        bias_initializer_value = -np.log((1 - np.array(0.01)) / np.array(0.01))
        self.output_conv_1 = layers.norm_act_conv_pool_upsample(num_filters=1, prior_filter_num=self.filter_num_per_level[0], kernel_size=1, use_norm_act=True, use_pooling=False, use_upsample=False, use_bias=True, bias_initializer=tf.constant_initializer(bias_initializer_value), negative_slope=self.negative_slope, norm=self.norm)
        if self.deep_supervision == True:
            self.output_conv_2 = layers.norm_act_conv_pool_upsample(num_filters=1, prior_filter_num=self.filter_num_per_level[1], kernel_size=1, use_norm_act=True, use_pooling=False, use_upsample=False, use_bias=True, bias_initializer=tf.constant_initializer(bias_initializer_value), negative_slope=self.negative_slope, norm=self.norm)
            self.output_conv_3 = layers.norm_act_conv_pool_upsample(num_filters=1, prior_filter_num=self.filter_num_per_level[2], kernel_size=1, use_norm_act=True, use_pooling=False, use_upsample=False, use_bias=True, bias_initializer=tf.constant_initializer(bias_initializer_value), negative_slope=self.negative_slope, norm=self.norm)
    #
    def call(self, x, training=True):
        #encoder
        skip_1 = self.encoder_layer_1(x)
        x = self.pooling_layer_1(skip_1)
        #
        skip_2 = self.encoder_layer_2(x)
        x = self.pooling_layer_2(skip_2)
        #
        skip_3 = self.encoder_layer_3(x)
        x = self.pooling_layer_3(skip_3)
        #
        skip_4 = self.encoder_layer_4(x)
        x = self.pooling_layer_4(skip_4)
        #
        skip_5 = self.encoder_layer_5(x)
        x = self.pooling_layer_5(skip_5)
        #
        #bottleneck
        out_6 = self.bottleneck(x)
        #
        #decoder
        x = self.upsampling_layer_5(out_6)
        x = tf.concat([x, skip_5], axis=-1)
        out_5 = self.decoder_layer_5(x)
        #
        x = self.upsampling_layer_4(out_5)
        x = tf.concat([x, skip_4], axis=-1)
        out_4 = self.decoder_layer_4(x)
        #
        x = self.upsampling_layer_3(out_4)
        x = tf.concat([x, skip_3], axis=-1)
        out_3 = self.decoder_layer_3(x)
        #
        x = self.upsampling_layer_2(out_3)
        x = tf.concat([x, skip_2], axis=-1)
        out_2 = self.decoder_layer_2(x)
        #
        x = self.upsampling_layer_1(out_2)
        x = tf.concat([x, skip_1], axis=-1)
        out_1 = self.decoder_layer_1(x)
        #
        #output layers / deep supervision
        out_1 = self.output_conv_1(out_1)
        if training == True and self.deep_supervision == True:
            out_2 = self.output_conv_2(out_2)
            out_3 = self.output_conv_3(out_3)
            out_1 = [out_1, out_2, out_3]
        #
        return out_1