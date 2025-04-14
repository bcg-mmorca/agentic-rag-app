// Interface definition
interface UserData {
    id: string;
    name: string;
    role: 'admin' | 'user';
}

// Enum example
enum TransactionStatus {
    PENDING = 'PENDING',
    COMPLETED = 'COMPLETED',
    FAILED = 'FAILED'
}

// Class with decorators
@Injectable()
class DataService {
    private static instance: DataService;
    #privateField: string;

    constructor() {
        this.#privateField = 'private value';
    }

    // Singleton pattern
    public static getInstance(): DataService {
        if (!DataService.instance) {
            DataService.instance = new DataService();
        }
        return DataService.instance;
    }

    // Async method with try-catch
    async fetchUserData(userId: string): Promise<UserData> {
        try {
            const response = await fetch(`/api/users/${userId}`);
            if (!response.ok) throw new Error('User not found');
            return await response.json();
        } catch (error) {
            console.error(`Error fetching user: ${error.message}`);
            throw error;
        }
    }

    // Generator function
    *idGenerator(): Generator<string> {
        let index = 0;
        while (true) {
            yield `id-${index++}`;
        }
    }
}

// Higher-order function
const memoize = (fn: Function) => {
    const cache = new Map();
    return (...args: any[]) => {
        const key = JSON.stringify(args);
        if (cache.has(key)) return cache.get(key);
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };
};

// React component with hooks
const UserComponent: React.FC<{ userId: string }> = ({ userId }) => {
    const [user, setUser] = useState<UserData | null>(null);
    const [loading, setLoading] = useState<boolean>(true);

    useEffect(() => {
        const loadUser = async () => {
            try {
                const data = await DataService.getInstance().fetchUserData(userId);
                setUser(data);
            } finally {
                setLoading(false);
            }
        };
        loadUser();
    }, [userId]);

    return loading ? <Spinner /> : (
        <div className="user-card">
            <h2>{user?.name}</h2>
            <span>{user?.role}</span>
        </div>
    );
};

// Event handling and prototypes
class EventEmitter {
    private events: Map<string, Function[]>;

    constructor() {
        this.events = new Map();
    }

    on(event: string, callback: Function): void {
        if (!this.events.has(event)) {
            this.events.set(event, []);
        }
        this.events.get(event)?.push(callback);
    }

    emit(event: string, data?: any): void {
        this.events.get(event)?.forEach(cb => cb(data));
    }
}

// Proxy pattern
const userProxy = new Proxy({}, {
    get: (target: any, prop: string) => {
        return `Accessing property: ${prop}`;
    },
    set: (target: any, prop: string, value: any) => {
        console.log(`Setting ${prop} = ${value}`);
        target[prop] = value;
        return true;
    }
});

// Module pattern
const Calculator = (() => {
    let result = 0;

    return {
        add: (x: number): void => { result += x; },
        subtract: (x: number): void => { result -= x; },
        getResult: (): number => result
    };
})();

// Error handling with custom error
class ValidationError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'ValidationError';
    }
}

// Async iterator
async function* asyncGenerator() {
    let i = 0;
    while (i < 3) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        yield i++;
    }
}

// Export statement
export {
    DataService,
    UserComponent,
    EventEmitter,
    ValidationError,
    TransactionStatus,
    Calculator
};
