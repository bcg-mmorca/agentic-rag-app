/**
 * Calculates how many times a specific day of week occurs on a specific date in a year
 * @param {number} year - The year to check
 * @param {number} dayOfMonth - The day of month (1-31)
 * @param {number} targetDayOfWeek - The day of week (0-6, where 0 is Sunday)
 * @returns {number} Count of occurrences
 */
function countSpecificDayInYear(year, dayOfMonth, targetDayOfWeek) {
    // Input validation
    if (dayOfMonth < 1 || dayOfMonth > 31) {
        throw new Error('Day of month must be between 1 and 31');
    }
    if (targetDayOfWeek < 0 || targetDayOfWeek > 6) {
        throw new Error('Day of week must be between 0 (Sunday) and 6 (Saturday)');
    }

    let count = 0;
    
    // Check each month of the year
    for (let month = 0; month < 12; month++) {
        // Create date object for the specific day in each month
        const date = new Date(year, month, dayOfMonth);
        
        // Check if the date is valid (handles months with less than 31 days)
        if (date.getDate() === dayOfMonth) {
            // Check if it falls on our target day of week
            if (date.getDay() === targetDayOfWeek) {
                count++;
            }
        }
    }
    
    return count;
}

/**
 * Helper function to convert day name to number
 * @param {string} dayName - Name of the day (e.g., "Sunday")
 * @returns {number} Day number (0-6)
 */
function getDayNumber(dayName) {
    const days = {
        'sunday': 0,
        'monday': 1,
        'tuesday': 2,
        'wednesday': 3,
        'thursday': 4,
        'friday': 5,
        'saturday': 6
    };
    return days[dayName.toLowerCase()];
}

/**
 * Counts occurrences of a specific day across a range of years
 * @param {number} startYear - Starting year
 * @param {number} endYear - Ending year (inclusive)
 * @param {number} dayOfMonth - The day of month (1-31)
 * @param {number} targetDayOfWeek - The day of week (0-6, where 0 is Sunday)
 * @returns {Object} Object containing total count and yearly breakdown
 */
function countSpecificDayInYearRange(startYear, endYear, dayOfMonth, targetDayOfWeek) {
    if (startYear > endYear) {
        throw new Error('Start year must be less than or equal to end year');
    }

    let result = 0;

    for (let year = startYear; year <= endYear; year++) {
        const count = countSpecificDayInYear(year, dayOfMonth, targetDayOfWeek);
        result += count;
    }

    return result;
}

/**
 * Generates a random Ethereum address with a prefix of a given human name
 * @param {string} name human name to prefix the address
 * @returns string
 */
function randomHumanNamedAddress(name) {
  const charset = 'abcdef0123456789';

  let address = '';
  for (let i = 0; i < 40 - name.length; i++) {
    address += charset[Math.floor(Math.random() * charset.length)];
  }

  return '0x' + name + address;
}

function generateBitcoinAddress() {
  const charset = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';

  let address = '';
  for (let i = 0; i < 34; i++) {
    address += charset[Math.floor(Math.random() * charset.length)];
  }

  return address;
}

/**
 * Calucaltes the power of a number
 */
function power(base, exponent) {
    if (exponent === 0) {
        return 1;
    }

    if (base === 0) {
      return 0;
    }

    return Math.pow(base, exponent);
}

// Example usage with test cases
function runTests() {
    console.log('Running tests...');
    
    // Test case 1: Count Friday the 13ths in 2024
    const fridayCount = countSpecificDayInYear(2024, 13, 5);
    console.log('Number of Friday the 13ths in 2024:', fridayCount);
    
    // Test case 2: Count Sunday the 1sts in 2024
    const sundayCount = countSpecificDayInYear(2024, 1, 0);
    console.log('Number of Sundays falling on the 1st of the month in 2024:', sundayCount);
    
    // Test with string day name
    const dayName = 'Sunday';
    const dayNum = getDayNumber(dayName);
    const result = countSpecificDayInYear(2024, 13, dayNum);
    console.log(`Number of ${dayName} the 13ths in 2024:`, result);
    
    // Error handling test
    try {
        countSpecificDayInYear(2024, 32, 0);
    } catch (error) {
        console.error('Expected error:', error.message);
    }

    // Test case for year range
    console.log('\nTesting year range functionality:');
    const rangeResults = countSpecificDayInYearRange(2024, 2026, 13, 5); // Friday the 13ths
    console.log('Friday the 13ths from 2024 to 2026:');
    console.log('Total count:', rangeResults.total);
    console.log('Yearly breakdown:', rangeResults.yearlyCount);
}

// Run tests if file is executed directly
if (require.main === module) {
    console.log(countSpecificDayInYear(2020, 7, 4))
    console.log(countSpecificDayInYearRange(2025, 2030, 5, 4))
    // runTests();
}
