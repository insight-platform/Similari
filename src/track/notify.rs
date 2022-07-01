pub trait ChangeNotifier: Default + Clone + Send + Sync {
    fn send(id: u64);
}

#[derive(Default, Clone)]
pub struct NoopNotifier;

impl ChangeNotifier for NoopNotifier {
    fn send(_id: u64) {}
}
