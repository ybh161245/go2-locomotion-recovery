"""Stratified normal/pressure lanes over the unchanged round051 mass task."""
from go2_mass_course_env import Go2MassCourseEnv, mass_assignment, course_events
from go2_sequence_env import build_schedule


def _pressure_workers(config):
    workers = tuple(int(worker) for worker in config['exposure_sampling']['pressure_workers'])
    if len(workers) != len(set(workers)):
        raise ValueError('Pressure worker IDs must be unique')
    if any(worker < 0 or worker >= int(config['workers']) for worker in workers):
        raise ValueError('Pressure worker ID is outside the vector environment')
    return workers


def build_exposure_schedule(config, worker, episode):
    """Keep full normal sequences and the old12s single-push task in fixed lanes."""
    course = config['sequence_course']
    if int(worker) in _pressure_workers(config):
        schedule = [dict(start_step=0, end_step=int(course['straight_slot_s']) * 50,
                         command=[.2, 0., 0.], phase='pressure-lane')]
        return schedule, int(course['normal_slots'])
    # Force a normal schedule without changing the original command RNG, worker,
    # episode, first-turn sign, or any globally shared configuration object.
    normal_config = dict(config, sequence_course=dict(course, normal_slots=course['total_slots']))
    return build_schedule(normal_config, int(worker), int(episode))


def pressure_events(config, worker, episode, options=None):
    """The old independent single-push stream; explicit events retain precedence."""
    if options is not None and 'events' in options:
        return course_events(config, worker, episode, None, options)
    if int(worker) not in _pressure_workers(config):
        return []
    return course_events(config, worker, episode, config['sequence_course']['normal_slots'])


class Go2ExposureCourseEnv(Go2MassCourseEnv):
    """Allocate data steps by worker lane; retain051 physics, reward, and mass RNG."""
    def __init__(self, command_mode='mixed', **kwargs):
        self.exposure_training = command_mode != 'external'
        # Use the parent's external initialization to avoid its episode-slot mix.
        super().__init__('external', **kwargs)
        _pressure_workers(self.config)
        self.use_sequence = self.exposure_training

    def reset(self, *, seed=None, options=None):
        episode = self.episode + 1
        scale, mass_meta = mass_assignment(self.config, self.worker, episode,
            external=not self.exposure_training, options=options)
        opts = dict(options or {})
        if self.exposure_training:
            schedule, slot = build_exposure_schedule(self.config, self.worker, episode)
            events = pressure_events(self.config, self.worker, episode, options)
            self.episode_seconds = schedule[-1]['end_step'] * .02
            assert int(self.episode_seconds / .02) == schedule[-1]['end_step']
            opts.update(command=schedule[0]['command'], events=events, mass_scale=scale)
        else:
            schedule, slot = [], None
            events = pressure_events(self.config, self.worker, episode,
                                     dict(opts, events=opts.get('events', [])))
            # External preflight is nominal unless explicitly overridden by caller.
            opts.update(events=events, mass_scale=scale)

        # The parent flushes the previous trace/metadata before replacing it.
        # Temporarily bypass only its episode-slot schedule; all mass/model reset,
        # force clearing, trace creation, and physical checks remain native051.
        self.use_sequence = False
        try:
            _, info = super().reset(seed=seed, options=opts)
        finally:
            self.use_sequence = self.exposure_training
        self.schedule = schedule
        self.events = events
        profile = ('pressure-lane' if self.worker in _pressure_workers(self.config)
                   else 'long-normal-lane') if self.exposure_training else 'explicit-preflight'
        self.episode_meta.update(mass_meta, profile=profile, slot=slot,
            command_schedule=schedule, episode_limit_steps=int(self.episode_seconds / .02),
            initial_command=self.command.tolist(), events=events,
            exposure_lane=profile, pressure_worker_ids=list(_pressure_workers(self.config)))
        return self.observe(), info
