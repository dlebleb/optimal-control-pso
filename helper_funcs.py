from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt


def animate_trajectory(X_full_best, params, interval=300):
    fig, ax = plt.subplots(figsize=(8, 8))

    p1_traj = X_full_best[:, 0:2]
    p2_traj = X_full_best[:, 2:4]

    # Eksen sınırları
    all_x = np.concatenate([p1_traj[:, 0], p2_traj[:, 0]])
    all_y = np.concatenate([p1_traj[:, 1], p2_traj[:, 1]])
    margin = 0.5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Trajektori Animasyonu")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Başlangıç noktaları (sabit)
    ax.plot(p1_traj[0, 0], p1_traj[0, 1], 'bs', markersize=10, label='p1 start')
    ax.plot(p2_traj[0, 0], p2_traj[0, 1], 'rs', markersize=10, label='p2 start')

    # d_encounter çemberi (p2 etrafında, sabit referans)
    d = params["d_encounter"]
    theta_circle = np.linspace(0, 2 * np.pi, 100)

    # Dinamik objeler
    line1,  = ax.plot([], [], 'b-',  linewidth=1.5, alpha=0.5)           # p1 izi
    line2,  = ax.plot([], [], 'r-',  linewidth=1.5, alpha=0.5)           # p2 izi
    dot1,   = ax.plot([], [], 'bo',  markersize=12, label='p1 wheelchair')
    dot2,   = ax.plot([], [], 'ro',  markersize=12, label='p2 obstacle')
    circle, = ax.plot([], [], 'r--', linewidth=1,   alpha=0.4, label='d_encounter')
    
    # Hız okları
    arrow1 = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    arrow2 = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='red', lw=2))

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11)
    ax.legend(loc='upper right')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        dot1.set_data([], [])
        dot2.set_data([], [])
        circle.set_data([], [])
        time_text.set_text('')
        return line1, line2, dot1, dot2, circle, time_text

    def update(frame):
        # İzler
        line1.set_data(p1_traj[:frame+1, 0], p1_traj[:frame+1, 1])
        line2.set_data(p2_traj[:frame+1, 0], p2_traj[:frame+1, 1])

        # Mevcut pozisyon
        dot1.set_data([p1_traj[frame, 0]], [p1_traj[frame, 1]])
        dot2.set_data([p2_traj[frame, 0]], [p2_traj[frame, 1]])

        # d_encounter çemberi p2 etrafında
        cx, cy = p2_traj[frame, 0], p2_traj[frame, 1]
        circle.set_data(cx + d * np.cos(theta_circle),
                        cy + d * np.sin(theta_circle))

        # Hız okları (ölçekli)
        scale = 0.8
        v1 = X_full_best[frame, 4:6]
        v2 = X_full_best[frame, 6:8]

        arrow1.set_position(p1_traj[frame])
        arrow1.xy = p1_traj[frame] + scale * v1

        arrow2.set_position(p2_traj[frame])
        arrow2.xy = p2_traj[frame] + scale * v2

        # Zaman + mesafe
        r = np.linalg.norm(p1_traj[frame] - p2_traj[frame])
        t = frame * params["dt"]
        time_text.set_text(f't = {t:.2f}s  |  r = {r:.2f}m')

        return line1, line2, dot1, dot2, circle, time_text, arrow1, arrow2

    anim = FuncAnimation(
        fig,
        update,
        frames=len(X_full_best),
        init_func=init,
        interval=interval,   # ms cinsinden — 300ms = yavaş, 100ms = hızlı
        blit=False
    )

    plt.tight_layout()
    plt.show()
    return anim

# Çağır
