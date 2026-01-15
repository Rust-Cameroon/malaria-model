use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct MalariaCNN<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B>,
    pool1: MaxPool2d,
    pool2: MaxPool2d,
    pool3: MaxPool2d,
    adaptive_pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc_species: Linear<B>,
    fc_stage: Linear<B>,
    relu: Relu,
    stage_loss_lambda: f32,
}

impl<B: Backend> MalariaCNN<B> {
    pub fn new(
        device: &B::Device,
        image_channels: usize,
        conv1_filters: usize,
        conv2_filters: usize,
        conv3_filters: usize,
        fc1_units: usize,
        fc2_units: usize,
        num_species_classes: usize,
        num_stage_classes: usize,
        stage_loss_lambda: f32,
        dropout_rate: f64,
    ) -> Self {
        let conv1 = Conv2dConfig::new([image_channels, conv1_filters], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let conv2 = Conv2dConfig::new([conv1_filters, conv2_filters], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let conv3 = Conv2dConfig::new([conv2_filters, conv3_filters], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let bn1 = BatchNormConfig::new(conv1_filters).init(device);
        let bn2 = BatchNormConfig::new(conv2_filters).init(device);
        let bn3 = BatchNormConfig::new(conv3_filters).init(device);

        let pool1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        let pool2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        let pool3 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let adaptive_pool = AdaptiveAvgPool2dConfig::new([4, 4]).init();
        let dropout = DropoutConfig::new(dropout_rate).init();

        let fc_input_size = conv3_filters * 4 * 4;
        let fc1 = LinearConfig::new(fc_input_size, fc1_units).init(device);
        let fc2 = LinearConfig::new(fc1_units, fc2_units).init(device);
        let fc_species = LinearConfig::new(fc2_units, num_species_classes).init(device);
        let fc_stage = LinearConfig::new(fc2_units, num_stage_classes).init(device);

        let relu = Relu::new();

        Self {
            conv1, bn1, conv2, bn2, conv3, bn3,
            pool1, pool2, pool3, adaptive_pool,
            dropout, fc1, fc2, fc_species, fc_stage, relu,
            stage_loss_lambda,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.pool1.forward(self.relu.forward(self.bn1.forward(self.conv1.forward(x))));
        let x = self.pool2.forward(self.relu.forward(self.bn2.forward(self.conv2.forward(x))));
        let x = self.pool3.forward(self.relu.forward(self.bn3.forward(self.conv3.forward(x))));
        let x = self.adaptive_pool.forward(x);
        let x = x.flatten(1, 3);
        let x = self.relu.forward(self.dropout.forward(self.fc1.forward(x)));
        let x = self.relu.forward(self.dropout.forward(self.fc2.forward(x)));
        let species = self.fc_species.forward(x.clone());
        let stages = self.fc_stage.forward(x);
        (species, stages)
    }
}

fn main() {}