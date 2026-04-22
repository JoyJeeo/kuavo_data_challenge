import torch
from torch import Tensor, nn

from kuavo_train.wrapper.policy.bc.BCConfigWrapper import CustomBCConfigWrapper


class CustomBCModelWrapper(nn.Module):
    """A simple state-only MLP BC model.

    Input:
      - observation.state: (B, Ds) or (B, S, Ds)
    Output:
      - predicted action: (B, A) or (B, n_action_steps, A)
    """

    def __init__(self, config: CustomBCConfigWrapper):
        super().__init__()
        self.config = config

        state_shape = self.config.robot_state_feature.shape
        action_shape = self.config.action_feature.shape

        # Flatten state across temporal dimension if present.
        self.state_dim = 1
        for d in state_shape:
            self.state_dim *= d

        # action feature in this repo is expected to be 1D: (A,)
        self.action_dim = int(action_shape[0])

        output_dim = self.action_dim * int(self.config.n_action_steps)

        layers: list[nn.Module] = []
        in_dim = self.state_dim
        for i in range(self.config.num_layers - 1):
            layers.append(nn.Linear(in_dim, self.config.hidden_dim))
            layers.append(nn.ReLU(inplace=False))
            if self.config.dropout > 0:
                layers.append(nn.Dropout(p=self.config.dropout))
            in_dim = self.config.hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch["observation.state"]
        if state.dim() == 3:
            state = state.flatten(start_dim=1)  # (B, S*Ds)
        elif state.dim() == 2:
            pass  # (B, Ds)
        else:
            raise ValueError(f"Unsupported state dim={state.dim()}, expected 2 or 3.")

        actions = self.net(state)

        if self.config.n_action_steps > 1:
            actions = actions.view(actions.shape[0], self.config.n_action_steps, self.action_dim)
        return actions
