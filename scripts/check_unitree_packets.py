"""Exercise real Unitree IDL serialization with archived encoder positions; no DDS I/O."""
import argparse
import json
from pathlib import Path

import numpy as np
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__HandState_ as new_hand_state,
    unitree_hg_msg_dds__LowState_ as new_body_state,
)

from movens_sim2real.unitree import G1Packets


def check(source, calibration=None):
    with np.load(source, allow_pickle=False) as a:
        j=len(a["q"])//2
        q = a["q"][j]
        if calibration:
            order=calibration["sdk_joint_names"]
            names=a["joint_names"].tolist()
            q=np.asarray(calibration["signs"])*q[[names.index(n) for n in order]]+calibration["zero_rad"]
            target=a["sdk_target43"][j]
        else:
            target=q
    if calibration and np.array_equal(q,target):
        raise ValueError("The actual proposal test requires a target other than the encoder sample")
    # Test channel order and skipped slots independently of a real device's calibration.
    body_slots = list(reversed(range(6, 35)))
    sdk = G1Packets(body_slots, dict(kp=[5.]*43, kd=[.3]*43), mode_machine=1, mode_pr=0)
    body, left, right = new_body_state(), new_hand_state(), new_hand_state()
    body.mode_machine = 1
    for bank, packet, offset in zip(sdk.channels, (body, left, right), (0, 29, 36)):
        for j, i in enumerate(bank):
            packet.motor_state[i].q = float(q[offset+j])
    body.crc = sdk.crc.Crc(body)
    states = [type(p).deserialize(p.serialize()) for p in (body, left, right)]
    q_received = np.concatenate([sdk.decode(i, p, 1.)[1]["q"] for i, p in enumerate(states)])
    commands = [type(p).deserialize(p.serialize()) for p in sdk.encode(
        target, states[0], captured_s=1., now_s=1.01, max_age_s=.03)]
    q_sent = np.array([p.motor_cmd[i].q for p, bank in zip(commands, sdk.channels) for i in bank])
    motor_modes = [[p.motor_cmd[i].mode for i in bank] for p, bank in zip(commands, sdk.channels)]
    if (not np.allclose(q,q_received,rtol=0.,atol=1e-7) or not np.allclose(target,q_sent,rtol=0.,atol=1e-7)
            or any(m.mode for m in commands[0].motor_cmd[:6])
            or motor_modes != [[1]*29, list(range(16, 23)), list(range(16, 23))]
            or sdk.crc.Crc(commands[0]) != commands[0].crc):
        raise ValueError("43-channel Unitree command or encoder round trip changed")
    try:
        sdk.encode(q_received, states[0], captured_s=1., now_s=1.08, max_age_s=.03)
    except ValueError as e:
        if "fresh calibrated" not in str(e):
            raise
    else:
        raise ValueError("A stale body state produced a wire packet")
    states[0].crc ^= 1
    try:
        sdk.decode(0, states[0], 1.)
    except ValueError as e:
        if "CRC" not in str(e):
            raise
    else:
        raise ValueError("A corrupt body state was accepted")
    return dict(input=str(source), domain="offline SDK IDL using archived positions and test calibration",
                actual_policy_proposal_serialized=calibration is not None,
                sdk_state_packets=3, sdk_command_packets=3, roundtrip_max_rad=float(max(abs(target-q_sent))),
                crc_verified=True, stale_and_corrupt_rejected=True, published=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--calibration",type=Path,help="Wire calibration when source comes from check_robot_control.py")
    a = p.parse_args()
    a.out.write_text(json.dumps(check(a.source,json.loads(a.calibration.read_text()) if a.calibration else None),indent=2)+"\n")
