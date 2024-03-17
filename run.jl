using CSV,DataFrames,MLDataUtils,Flux,Statistics

data = CSV.read("data/dataset.csv", DataFrame)
data.class = ifelse.(data.class .== "s", 0, 1)

X = Matrix(data[:, 1:3])
Y = Float32.(data.class)

(train_indices, val_test_indices) = splitobs(shuffleobs(1:length(Y)), at=0.7)
(val_indices, test_indices) = splitobs(val_test_indices, at=0.666)

X_train, Y_train = X[train_indices, :], Y[train_indices]
X_val, Y_val = X[val_indices, :], Y[val_indices]
X_test, Y_test = X[test_indices, :], Y[test_indices]

for i in 1:size(X_train, 2)
    col_mean = mean(X_train[:, i])
    col_std = std(X_train[:, i])
    X_train[:, i] = (X_train[:, i] .- col_mean) ./ col_std
    X_val[:, i] = (X_val[:, i] .- col_mean) ./ col_std
    X_test[:, i] = (X_test[:, i] .- col_mean) ./ col_std
end

Y_train_onehot = Flux.onehotbatch(Y_train, 0:1)
Y_val_onehot = Flux.onehotbatch(Y_val, 0:1)
Y_test_onehot = Flux.onehotbatch(Y_test, 0:1)

batch_size = 128

train_dataloader = Flux.DataLoader((X_train', Y_train_onehot), batchsize=batch_size, shuffle=true)
val_dataloader = Flux.DataLoader((X_val', Y_val_onehot), batchsize=batch_size, shuffle=false)
test_dataloader = Flux.DataLoader((X_test', Y_test_onehot), batchsize=batch_size, shuffle=false)


function loss(x, y)
    ŷ = model(x)
    return Flux.logitcrossentropy(ŷ, y)
end

function accuracy(data_loader, model)
    correct = 0
    total = 0
    for (x, y) in data_loader
        predictions = model(x)
        correct_preds = argmax(predictions, dims=1) .== argmax(y, dims=1)
        correct += sum(correct_preds)
        total += length(y)
    end
    correct / total
end

model = Chain(
  Dense(3, 32, relu),
  Dense(32, 16, relu), 
  Dense(16, 2) 
)

optimizer = AdaGrad()
epochs = 100

for epoch in 1:epochs
    for (x, y) in train_dataloader
        grads = Flux.gradient(Flux.params(model)) do
            train_loss = loss(x, y)
            train_loss
        end
        Flux.Optimise.update!(optimizer, Flux.params(model), grads)
    end

    train_acc = accuracy(train_dataloader, model)
    val_acc = accuracy(val_dataloader, model)
    println("Epoch $epoch: Training Accuracy: $train_acc, Validation Accuracy: $val_acc")
end

test_acc = accuracy(test_dataloader, model)
println("Test Accuracy: $test_acc")