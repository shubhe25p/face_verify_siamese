class ConvSiameseNet(nn.Module):
  """
  Convolutional Siamese Network from "Siamese Neural Networks for One-shot Image Recognition"
  Paper can be found at http://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf
  """

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 64, 10)
    self.conv2 = nn.Conv2d(64, 128, 7)
    self.conv3 = nn.Conv2d(128, 128, 4)
    self.conv4 = nn.Conv2d(128, 256, 4)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(256*6*6, 4096)
    self.fc2 = nn.Linear(4096, 1)

  def model(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = F.relu(self.conv4(x))
    x = torch.flatten(x, 1)
    x = torch.sigmoid(self.fc1(x))
    return x

  def forward(self, x1, x2):
    x1_fv = self.model(x1)
    x2_fv = self.model(x2)
    # Calculate L1 distance (as l1_distance) between x1_fv and x2_fv
    l1_distance = torch.abs(x1_fv - x2_fv)

    return self.fc2(l1_distance)

train_dataset, val_dataset = get_train_val_datasets(background_dataset_size=10000, val_split=0.2, download=False)

batch_size = 16

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g_seed
    )

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g_seed
    )

siamese_net = ConvSiameseNet()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(siamese_net.parameters(), lr=0.0001)

# Change this to train for any number of epochs
num_epochs = 10

for epoch in range(num_epochs):
  print(f"\n Epoch {epoch + 1} / {num_epochs} ================================")

  train_siamese_network(siamese_net, criterion, optimizer, train_loader, DEVICE)
  evaluate_siamese_network(siamese_net, criterion, val_loader, DEVICE)

