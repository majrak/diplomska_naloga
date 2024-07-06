on run {targetBuddyPhone, targetMessage, imagePath}
    set image to POSIX file imagePath
    tell application "Messages"
        set targetService to 1st service whose service type = iMessage
        set targetBuddy to buddy targetBuddyPhone of targetService

        send file image to targetBuddy
        send targetMessage to targetBuddyPhone
    end tell
end run